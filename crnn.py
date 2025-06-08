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
        datalayer: nn.Module = None,
    ):
        super().__init__()
        self.num_cascades = num_cascades
        self.chans = chans
        self.datalayer = datalayer
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

        # Create a list of DC layers, one for each cascade
        dcs = []
        for i in range(self.num_cascades):
            dcs.append(self.datalayer)
        self.dcs = nn.ModuleList(dcs)  # Use nn.ModuleList to register modules correctly

    def forward(self, x_init_permuted, y, mask):
        x = rearrange(x_init_permuted.clone(), "b h w t c -> b c h w t").float()
        b, ch, w, h, t = x.size()
        size_h = [t * b, self.chans, w, h]
        net = {}
        rcnn_layers = 6
        # Initialize hidden states for the first cascade (t0)
        for j in range(rcnn_layers - 1):
            net[f"t0_x{j}"] = torch.zeros(size_h).to(x.device)

        # This `x_cascades` will be the input to the recurrent layers at each cascade
        x_cascades = x

        for i in range(1, self.num_cascades + 1):
            # --- Recurrent Block ---
            x_rnn_in = rearrange(x_cascades, "b c h w t -> t b c h w").contiguous()
            # The hidden state for the RNN is always t0_x0, which is zeros.
            # This means the RNN state is reset at each cascade. This is a design choice.
            net[f"t{i}_x0"] = self.bcrnn(x_rnn_in, net["t0_x0"])
            net[f"t{i}_x1"] = self.bcrnn2(x_rnn_in, net[f"t{i}_x0"])

            # --- Convolutional Block with Recurrence over Cascades ---
            # Reshape for 2D convolutions
            conv_in = net[f"t{i}_x1"].view(-1, self.chans, w, h)

            # Layer 2
            x2 = self.conv1_x(conv_in)
            h2 = self.conv1_h(net[f"t{i - 1}_x2"])  # Use previous cascade's x2
            net[f"t{i}_x2"] = self.relu(h2 + x2)

            # Layer 3
            x3 = self.conv2_x(net[f"t{i}_x2"])
            h3 = self.conv2_h(net[f"t{i - 1}_x3"])  # Use previous cascade's x3
            net[f"t{i}_x3"] = self.relu(h3 + x3)

            # Layer 4
            x4 = self.conv3_x(net[f"t{i}_x3"])
            h4 = self.conv3_h(net[f"t{i - 1}_x4"])  # Use previous cascade's x4
            net[f"t{i}_x4"] = self.relu(h4 + x4)

            # Output Layer
            x5 = self.conv4_x(net[f"t{i}_x4"])

            # Res-connection
            x_res = x_rnn_in.view(-1, ch, w, h)
            out_before_dc = x_res + x5

            # Reshape before the DC step
            out_permuted = rearrange(
                out_before_dc, "(b t) c h w -> b h w t c", b=b
            ).contiguous()

            # --- Data Consistency ---
            x_dc_permuted = self.dcs[i - 1](out_permuted, y, mask)

            # The output of the DC layer becomes the input for the *next* cascade's RNN block
            x_cascades = rearrange(x_dc_permuted, "b h w t c -> b c h w t")

        # The final output is the result from the last DC step
        return x_dc_permuted


class ArtifactRemovalCRNN(nn.Module):
    """This wrapper's role is now restored to its original purpose:
    preparing the initial image and passing ALL necessary data to the backbone."""

    def __init__(self, backbone_net: CRNN):
        super().__init__()
        self.backbone_net = backbone_net

    def forward(self, y: torch.Tensor, physics, **kwargs):
        if isinstance(physics, nn.DataParallel):
            physics = physics.module

        # Get initial aliased image from measurements
        x_init = physics.A_adjoint(y)
        x_init = rearrange(x_init, "b c t h w -> b h w t c")

        # The mask for the DC layer should have the same shape as y
        mask = torch.ones_like(y)

        # Pass the initial image, the k-space measurements, and the mask
        x_hat = self.backbone_net(x_init, y, mask)

        return rearrange(x_hat, "b h w t c -> b c t h w")
