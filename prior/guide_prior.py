import torch
from pyro.distributions import Normal
import pyro
import torch.nn.functional as F


def standard_normal_prior(name, params):
    mu_param = pyro.param('{}_mu'.format(name), torch.randn_like(params))
    sigma_param = F.softplus(pyro.param('{}_sigma'.format(name), torch.randn_like(params)))
    prior = Normal(loc=mu_param, scale=sigma_param)
    return prior


def mean_field_normal_prior(name, params, eps=10e-7, loc_init_std=0.1, scale_init_mean=-3, scale_init_std=0.1):
    loc_init = pyro.param('{}_mu'.format(name),
                          torch.normal(mean=torch.zeros_like(params),
                                       std=torch.mul(torch.ones_like(params), loc_init_std)))

    untransformed_scale_init = pyro.param('{}_sigma'.format(name),
                                          torch.normal(mean=torch.ones_like(params) * scale_init_mean,
                                                       std=torch.mul(torch.ones_like(params), scale_init_std)))
    sigma = eps + F.softplus(untransformed_scale_init)
    dist = Normal(loc=loc_init, scale=sigma)
    return dist