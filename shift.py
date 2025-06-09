import torch 
from deepinv.transform.temporal import ShiftTime

class ShiftTimeForward(ShiftTime):
    """
    Like ShiftTime but:

    • shifts only toward later frames (positive amounts)
    • never wraps – the new frames that “enter” are padded
      either with reflection (default) or with zeros.

    Parameters
    ----------
    n_trans   : int   # copies per input
    padding   : str   # "reflect" (default, same as original) or "zero"
    """
    def __init__(self, *args, padding="reflect", **kwargs):
        super().__init__(*args, padding=("reflect" if padding == "zero" else padding),
                         **kwargs)
        self.zero_pad = padding == "zero"     # remember if user asked for zeros

    # ------------------------------------------------------------------ #
    # 1. pick **positive** offsets only
    # ------------------------------------------------------------------ #
    def _get_params(self, x):
        T = x.shape[-3]
        # random integers 1 .. T-1  (strictly > 0)
        shifts = torch.randint(
            1, T, (self.n_trans,),
            generator=self.rng, device=x.device
        )
        return {"amounts": shifts}

    # ------------------------------------------------------------------ #
    # 2. apply shift with chosen padding
    # ------------------------------------------------------------------ #
    def _transform(self, x, amounts=[], **_):
        outs = []
        for s in amounts:
            s = int(s)
            if s == 0:
                outs.append(x)
                continue

            if self.zero_pad:                    # ---- zero-padding branch
                pad = torch.zeros_like(x[:, :s])
                shifted = torch.cat([pad, x[:, :-s]], dim=-3)
            else:                                # ---- reflect branch (no wrap)
                shifted = self.roll_reflect_1d(x, by=s, dim=-3)

            outs.append(shifted)

        return torch.cat(outs, dim=0)
