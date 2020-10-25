import torch


class RunningStats(object):
    """Computes running mean and standard deviation"""

    def __init__(self, device, dim, n=0., m=None, s=None):
        self.n = n
        self.m = m
        self.s = s
        self.dim = dim
        self.device = device

    def clear(self):
        self.n = 0.

    def push(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float, device=self.device)
        self.update_params(x)

    def update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = torch.zeros(self.dim, dtype=torch.float, device=self.device)
        else:
            prev_m = self.m.clone()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def __add__(self, other):
        if isinstance(other, RunningStats):
            return RunningStats(self.n+other.n, self.n+other.n, self.s+other.s)
        else:
            self.push(other)
            return self

    @property
    def mean(self):
        return self.m if self.n else torch.zeros(self.dim, dtype=torch.float, device=self.device)

    def variance(self):
        return self.s / self.n if self.n else torch.zeros(self.dim, dtype=torch.float, device=self.device)

    @property
    def std(self):
        return torch.sqrt(self.variance())

    def normalize(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float, device=self.device)
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float, device=self.device)
        return x * self.std + self.mean
