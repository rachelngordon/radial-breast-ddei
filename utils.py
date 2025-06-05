import deepinv as dinv
from einops import rearrange

class OverTime(dinv.transform.Transform):
    """
    Apply an arbitrary 2-D DeepInv transform to every time-frame
    of a 5-D tensor (B, C, T, H, W) → (B, C, T, H, W).
    """
    def __init__(self, inner, dim: int = 2):
        super().__init__()
        self.inner = inner
        self.dim = dim          # usually 2 for the time axis

    def _get_params(self, x):
        return {}

    def _transform(self, x, **params):
        # x: (B,C,T,H,W)
        b, c, t, h, w = x.shape
        x4 = rearrange(x, 'b c t h w -> (b t) c h w')
        y4 = self.inner(x4)

        return rearrange(y4, '(b t) c h w -> b c t h w', b=b)