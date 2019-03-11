import torch
from pyro.distributions import Normal


def normal_prior(params):
    dist = Normal(loc=torch.zeros_like(params), scale=torch.ones_like(params))
    return dist