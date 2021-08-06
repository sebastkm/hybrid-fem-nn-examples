import torch
import torch.nn as nn


class ForcingTerm(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(4, 16)
        self.lin1.weight.data.copy_(self.lin1.weight.data / 1000)
        self.lin2 = nn.Linear(16, 16)
        self.lin2.weight.data.copy_(self.lin2.weight.data / 1000)
        self.lin3 = nn.Linear(16, 16)
        self.lin3.weight.data.copy_(self.lin3.weight.data / 1000)
        self.lin4 = nn.Linear(16, 3)
        self.lin4.weight.data.copy_(self.lin4.weight.data / 1000)

        self.a = nn.Parameter(torch.tensor(1.))
        self.n = 1

    def forward(self, v, s):
        x_ = torch.cat((v, s), -1)
        x1 = self.lin1(x_)
        x = torch.tanh(self.n * self.a * x1)
        x2 = self.lin2(x)
        x_ = torch.tanh(self.n * self.a * x2)
        x3 = self.lin3(x_)
        x = torch.tanh(self.n * self.a * x3)
        x = self.lin4(x + x1)
        return x


class IonicCurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(4, 16)
        self.lin0.weight.data.copy_(self.lin0.weight.data / 1000)
        self.lin1 = nn.Linear(16, 16)
        self.lin1.weight.data.copy_(self.lin1.weight.data / 1000)
        self.lin2 = nn.Linear(16, 16)
        self.lin2.weight.data.copy_(self.lin2.weight.data / 1000)
        self.lin3 = nn.Linear(16, 1)
        self.lin3.weight.data.copy_(self.lin3.weight.data / 1000)

        self.a = nn.Parameter(torch.tensor(1.))
        self.n = 1

    def forward(self, v, s):
        x_ = torch.cat((v, s), -1)

        x0_ = self.lin0(x_)
        x0 = torch.tanh(self.n * self.a * x0_)

        x1_ = self.lin1(x0)
        x1 = torch.tanh(self.n * self.a * x1_)

        x2_ = self.lin2(x1)
        x2 = torch.tanh(self.n * self.a * x2_)

        x3 = self.lin3(x2 + x0_)
        return x3


class GTIonTerm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v, s):
        x_ = torch.cat((v, s), -1)

        V = x_[..., 0].unsqueeze(-1)
        m = x_[..., 1].unsqueeze(-1)
        h = x_[..., 2].unsqueeze(-1)
        n = x_[..., 3].unsqueeze(-1)

        # Assign parameters
        g_Na = 120.0
        g_K = 36.0
        g_L = 0.3
        Cm = 1.0
        E_R = -75.0

        # Expressions for the Sodium channel component
        E_Na = 115.0 + E_R
        i_Na = g_Na*(m*m*m)*(-E_Na + V)*h

        # Expressions for the Potassium channel component
        E_K = -12.0 + E_R
        i_K = g_K*n**4*(-E_K + V)

        # Expressions for the Leakage current component
        E_L = 10.613 + E_R
        i_L = g_L*(-E_L + V)

        # Expressions for the Membrane component
        # TODO can be defined outside
        i_Stim = 0.0
        return -(-i_K - i_L - i_Na + i_Stim)/Cm


class GTODETerm(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.))

    def forward(self, v, s):
        x_ = torch.cat((v, s), -1)

        V = x_[..., 0].unsqueeze(-1)
        m = x_[..., 1].unsqueeze(-1)
        h = x_[..., 2].unsqueeze(-1)
        n = x_[..., 3].unsqueeze(-1)

        # Init return args
        F_expressions = []

        # Expressions for the m gate component
        alpha_m = (-5.0 - 0.1*V)/(-1.0 + torch.exp(-5.0 - V/10.0))
        beta_m = 4*torch.exp(-25.0/6.0 - V/18.0)
        F_expressions.append((1 - m)*alpha_m - beta_m*m)

        # Expressions for the h gate component
        alpha_h = 0.07*torch.exp(-15.0/4.0 - V/20.0)
        beta_h = 1.0/(1 + torch.exp(-9.0/2.0 - V/10.0))
        F_expressions.append((1 - h)*alpha_h - beta_h*h)

        # Expressions for the n gate component
        alpha_n = (-0.65 - 0.01*V)/(-1.0 + torch.exp(-13.0/2.0 - V/10.0))
        beta_n = 0.125*torch.exp(-15.0/16.0 - V/80.0)
        F_expressions.append((1 - n)*alpha_n - beta_n*n)

        # Return results
        return torch.cat(F_expressions, -1)
