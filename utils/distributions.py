from torch.distributions import Distribution, Normal

class SoftNormal(Distribution):
    def __init__(self, loc, scale):
        super().__init__()
        self.normal = Normal(loc, scale)
        