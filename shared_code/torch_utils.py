import torch
import torch.autograd as autograd


def grad(u, x, retain_graph=True, create_graph=True, device=None):
    adj_inp = torch.ones(u.shape, device=device)
    r, *_ = autograd.grad(u, x, adj_inp, retain_graph=retain_graph, create_graph=create_graph)
    return r


def div(u, x, retain_graph=True, create_graph=True, device=None):
    """
        u: Shape [BATCH_SIZE, d, ...]
    """
    shape = u.shape
    dim = shape[1]
    r = 0.
    for i in range(dim):
        adj_inp = torch.zeros(shape, device=device)
        adj_inp[:, i, ...] = 1
        r_, *_ = autograd.grad(u, x, adj_inp, retain_graph=retain_graph, create_graph=create_graph)
        r += r_[:, i].unsqueeze(1)
    return r


def laplace(u, x):
    return div(grad(u, x), x)


def split_cartesian_prod(a, b):
    """Return the cartesian product split into two separate tensors.
    """
    ab = torch.cartesian_prod(a, b)
    a = ab[..., 0].unsqueeze(-1)
    b = ab[..., 1].unsqueeze(-1)
    return a, b
