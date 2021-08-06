import torch
from scipy.optimize import minimize
import numpy as np

from functools import reduce


class ScipyOptimizer:
    def __init__(self, parameters, method, maxiter, callback=lambda *args: None, **kwargs):
        self.kwargs = kwargs
        self.parameters = list(parameters)
        self.method = method
        self.maxiter = maxiter
        self.callback = callback

        self.param_groups = []

    def step(self, closure):
        def fun(x, *args):
            # Set weight values
            with torch.no_grad():
                offset = 0
                for p in self.parameters:
                    n = 1
                    if len(p.shape) > 0:
                        n = reduce(lambda x, y: x*y, p.shape)
                    slice = x[offset:offset + n]
                    p.copy_(torch.tensor(slice).reshape(p.shape))
                    offset += n

            # Run forward + backward
            return float(closure())

        def jac(x, *args):
            # Fetch backward result
            j = []
            for p in self.parameters:
                j.append(p.grad.flatten().detach().numpy())
            return np.concatenate(j)

        xs = []
        for p in self.parameters:
            xs.append(p.flatten().detach().numpy())
        x0 = np.concatenate(xs)

        options = self.kwargs.copy()
        options["maxiter"] = self.maxiter

        minimize(fun, x0, method=self.method, callback=self.callback, jac=jac, options=options)

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

