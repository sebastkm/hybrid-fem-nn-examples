import torch
import torch.nn as nn

from shared_code.torch_utils import grad, div


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.u = PDESolution()
        self.phi = PDESolution()
        self.NN = PDETerm()

    def forward(self, x, t):
        x.requires_grad = True

        u = self.u(x, t)
        phi = self.phi(x, t)
        E = self.NN(x)
        u_t = grad(u, t)

        # PDE
        loss = ((u_t - div(phi, x)) ** 2).mean()
        loss += ((phi - E * grad(u, x)) ** 2).mean()
        return loss

    def obs_loss(self, x, t, obs):
        # Observations
        loss = ((self.u(x, t) - obs)**2).mean()
        return loss

    def bc_loss(self, x, t):
        return ((self.u(x, t))**2).mean()

    def double(self):
        self.u.double()
        self.NN.double()
        return super().double()

    def ground_truth(self, x):
        one = torch.tensor(1.0)
        two = torch.tensor(2.0)
        return torch.where(x < 0.25, one, torch.where(x > 0.75, one, two))


class PDESolution(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(2, 16)
        self.lin2 = nn.Linear(16, 16)
        self.lin3 = nn.Linear(16, 16)
        self.lin4 = nn.Linear(16, 1)

    def forward(self, x, t):
        x = torch.cat((x, t), 1)
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.lin3(x)
        x = torch.tanh(x)
        x = self.lin4(x)
        return x


class PDETerm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1, 30)
        self.lin2 = nn.Linear(30, 1)

        for param in self.parameters():
            param.data.uniform_()

    def forward(self, x):
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        return x
