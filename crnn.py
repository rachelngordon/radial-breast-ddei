import torch
import torch.nn as nn
from einops import rearrange


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
    ):
        super(CRNNcell, self).__init__()
        self.i2h = nn.Conv2d(
            input_size, hidden_size, kernel_size, padding=kernel_size // 2
        )
        self.h2h = nn.Conv2d(
            hidden_size, hidden_size, kernel_size, padding=kernel_size // 2
        )
        self.ih2ih = nn.Conv2d(
            hidden_size, hidden_size, kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        input: torch.Tensor,
        hidden_iteration: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)
        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
    ):
        """
        Args:
            input_size: Number of input channels
            hidden_size: Number of RCNN hidden layers channels
            kernel_size: Size of convolutional kernel
        """
        super(BCRNNlayer, self).__init__()

        self.hidden_size = hidden_size
        self.CRNN_model = CRNNcell(input_size, self.hidden_size, kernel_size)

    def forward(
        self, input: torch.Tensor, hidden_iteration: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input: Input 5D tensor of shape `(t, b, ch, h, w)`
            hidden_iteration: hidden states (output of BCRNNlayer) from previous
                    iteration, 5d tensor of shape (t, b, hidden_size, h, w)
        Returns:
            Output tensor of shape `(t, b, hidden_size, h, w)`.
        """
        t, b, ch, h, w = input.shape
        size_h = [b, self.hidden_size, h, w]

        hid_init = torch.zeros(size_h).to(input.device)
        output_f = []
        output_b = []

        # forward
        hidden = hid_init
        for i in range(t):
            hidden = self.CRNN_model(input[i], hidden_iteration[i], hidden)
            output_f.append(hidden)
        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(t):
            hidden = self.CRNN_model(
                input[t - i - 1], hidden_iteration[t - i - 1], hidden
            )
            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if b == 1:
            output = output.view(t, 1, self.hidden_size, h, w)

        return output


class CRNN(nn.Module):
    def __init__(
        self,
        num_cascades: int = 10,
        chans: int = 64,
        # REMOVED: datalayer=DCLayer(),
    ):
        super().__init__()
        self.num_cascades = num_cascades
        self.chans = chans
        # REMOVED: self.datalayer = datalayer
        self.bcrnn = BCRNNlayer(input_size=2, hidden_size=self.chans, kernel_size=3)
        self.bcrnn2 = BCRNNlayer(input_size=2, hidden_size=self.chans, kernel_size=3)
        self.conv1_x = nn.Conv2d(self.chans, self.chans, 3, padding=3 // 2)
        self.conv1_h = nn.Conv2d(self.chans, self.chans, 3, padding=3 // 2)
        self.conv2_x = nn.Conv2d(self.chans, self.chans, 3, padding=3 // 2)
        self.conv2_h = nn.Conv2d(self.chans, self.chans, 3, padding=3 // 2)
        self.conv3_x = nn.Conv2d(self.chans, self.chans, 3, padding=3 // 2)
        self.conv3_h = nn.Conv2d(self.chans, self.chans, 3, padding=3 // 2)
        self.conv4_x = nn.Conv2d(self.chans, 2, 3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)
        # REMOVED: dcs = [] ... self.dcs = dcs

    def forward(self, x):  # MODIFIED: removed y and mask arguments
        """
        Args:
            x: Input image, shape (b, w, h, t, ch)
        Returns:
            x: Reconstructed image, shape (b, w, h, t, ch)
        """
        x = rearrange(x.clone(), "b h w t c -> b c h w t").float()
        b, ch, w, h, t = x.size()
        size_h = [t * b, self.chans, w, h]
        net = {}
        rcnn_layers = 6
        for j in range(rcnn_layers - 1):
            net["t0_x%d" % j] = torch.zeros(size_h).to(x.device)

        for i in range(1, self.num_cascades + 1):
            x = rearrange(x, "b c h w t -> t b c h w")
            x = x.contiguous()

            net["t%d_x0" % (i - 1)] = net["t%d_x0" % (i - 1)].view(
                t, b, self.chans, w, h
            )
            net["t%d_x0" % i] = self.bcrnn(x, net["t%d_x0" % (i - 1)])
            net["t%d_x1" % i] = self.bcrnn2(x, net["t%d_x0" % i])
            net["t%d_x1" % i] = net["t%d_x1" % i].view(-1, self.chans, w, h)
            net["t%d_x2" % i] = self.conv1_x(net["t%d_x1" % i])
            net["t%d_h2" % i] = self.conv1_h(net["t%d_x2" % (i - 1)])
            net["t%d_x2" % i] = self.relu(net["t%d_h2" % i] + net["t%d_x2" % i])
            net["t%d_x3" % i] = self.conv2_x(net["t%d_x2" % i])
            net["t%d_h3" % i] = self.conv2_h(net["t%d_x3" % (i - 1)])
            net["t%d_x3" % i] = self.relu(net["t%d_h3" % i] + net["t%d_x3" % i])
            net["t%d_x4" % i] = self.conv3_x(net["t%d_x3" % i])
            net["t%d_h4" % i] = self.conv3_h(net["t%d_x4" % (i - 1)])
            net["t%d_x4" % i] = self.relu(net["t%d_h4" % i] + net["t%d_x4" % i])
            net["t%d_x5" % i] = self.conv4_x(net["t%d_x4" % i])

            x = x.view(-1, ch, w, h)
            net["t%d_out" % i] = x + net["t%d_x5" % i]

            # --- MODIFICATION: The entire DC block is removed ---
            # The output of the network block is now directly the input for the next iteration.
            # We just need to reshape it correctly.
            # net["t%d_out" % i] is shape (-1, ch, w, h) which is (b*t, ch, w, h)
            x = rearrange(net["t%d_out" % i], "(b t) c h w -> b c h w t", b=b)

        out = x  # The output of the final loop is 'x'
        return rearrange(out, "b c h w t -> b h w t c")


class ArtifactRemovalCRNN(nn.Module):
    r"""
    Artifact removal architecture :math:`\phi(A^{\top}y)`.
    Performs pseudo-inverse to get zero filled x_u, then passes x_u to the network.
    The network now acts as a pure denoiser without an internal DC step.
    """

    def __init__(self, backbone_net: CRNN):
        super().__init__()
        self.backbone_net = backbone_net

    def forward(self, y: torch.Tensor, physics, **kwargs):
        r"""
        Reconstructs a signal estimate from measurements y
        :param Tensor y: measurements [B,C,T,S,I] for radial
        :param deepinv.physics.Physics physics: forward operator
        """
        if isinstance(physics, nn.DataParallel):
            physics = physics.module

        # Get initial aliased image from measurements
        x_init = physics.A_adjoint(y)  # Output shape: (b, c, t, h, w)

        # Rearrange image into the format CRNN expects: (b, h, w, t, c)
        x_init = rearrange(x_init, "b c t h w -> b h w t c")

        # MODIFIED: Call the backbone with only the initial image
        x_hat = self.backbone_net(x_init)  # Backbone output: (b, h, w, t, c)

        # Rearrange the final image back to the standard format: (b, c, t, h, w)
        return rearrange(x_hat, "b h w t c -> b c t h w")
