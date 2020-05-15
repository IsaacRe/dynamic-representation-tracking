# code from https://github.com/shaabhishek/gumbel-softmax-pytorch/blob/master/Categorical%20VAE.ipynb
import torch


def sample_gumbel(shape, device, eps=1e-20):
    unif = torch.rand(*shape).to(device)
    g = -torch.log(-torch.log(unif + eps))
    return g


def sample_gumbel_diff(shape, device, eps=1e-20):
    return sample_gumbel(shape, device, eps=eps) - sample_gumbel(shape, device, eps=eps)


def gumbel_softmax(logits, temperature, device=0):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar

        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    g = sample_gumbel(logits.shape, device)
    h = (g + logits) / temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y


def gumbel_softmax_binary(logits, temperature, device=0):
    g = sample_gumbel_diff(logits.shape, device)
    h = (g + logits) / temperature
    cache = torch.exp(h)
    y = cache / (cache + 1.)
    return y
