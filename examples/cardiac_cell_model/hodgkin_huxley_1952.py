"""
This module contains a Hodgkin_Huxley_1952 cardiac cell model
"""
from __future__ import division
from collections import OrderedDict
import ufl
from dolfin import as_vector, Expression
from dolfin_adjoint import Constant


class Hodgkin_Huxley_1952(object):
    def __init__(self, params=None, init_conditions=None):
        """
        Create cardiac cell model

        *Arguments*
         params (dict, :py:class:`dolfin.Mesh`, optional)
           optional model parameters
         init_conditions (dict, :py:class:`dolfin.Mesh`, optional)
           optional initial conditions
        """
        self._parameters = self.default_parameters()

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        params = OrderedDict([("g_Na", 120.0),
                              ("g_K", 36.0),
                              ("g_L", 0.3),
                              ("Cm", 1.0),
                              ("E_R", -75.0)])
        return params

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        ic = OrderedDict([("V", -75.0),
                          ("m", 0.052955),
                          ("h", 0.595994),
                          ("n", 0.317732)])
        return ic

    def _I(self, v, s, time):
        """
        Original gotran transmembrane current dV/dt
        """
        time = time if time else Constant(0.0)

        # Assign states
        V = v
        assert(len(s) == 3)
        m, h, n = s

        # Assign parameters
        g_Na = self._parameters["g_Na"]
        g_K = self._parameters["g_K"]
        g_L = self._parameters["g_L"]
        Cm = self._parameters["Cm"]
        E_R = self._parameters["E_R"]

        # Init return args
        current = [ufl.zero()]*1

        # Expressions for the Sodium channel component
        E_Na = 115.0 + E_R
        i_Na = g_Na*(m*m*m)*(-E_Na + V)*h

        # Expressions for the Potassium channel component
        E_K = -12.0 + E_R
        i_K = g_K*ufl.elem_pow(n, 4)*(-E_K + V)

        # Expressions for the Leakage current component
        E_L = 10.613 + E_R
        i_L = g_L*(-E_L + V)

        # Expressions for the Membrane component
        # TODO can be defined outside
        i_Stim = 0.0
        current[0] = (-i_K - i_L - i_Na + i_Stim)/Cm

        # Return results
        return current[0]

    def I(self, v, s, time=None):
        """
        Transmembrane current

           I = -dV/dt

        """
        return -self._I(v, s, time)

    def F(self, v, s, time=None):
        """
        Right hand side for ODE system
        """
        time = time if time else Constant(0.0)

        # Assign states
        V = v
        assert(len(s) == 3)
        m, h, n = s

        # Assign parameters

        # Init return args
        F_expressions = [ufl.zero()]*3

        # Expressions for the m gate component
        alpha_m = (-5.0 - 0.1*V)/(-1.0 + ufl.exp(-5.0 - V/10.0))
        beta_m = 4*ufl.exp(-25.0/6.0 - V/18.0)
        F_expressions[0] = (1 - m)*alpha_m - beta_m*m

        # Expressions for the h gate component
        alpha_h = 0.07*ufl.exp(-15.0/4.0 - V/20.0)
        beta_h = 1.0/(1 + ufl.exp(-9.0/2.0 - V/10.0))
        F_expressions[1] = (1 - h)*alpha_h - beta_h*h

        # Expressions for the n gate component
        alpha_n = (-0.65 - 0.01*V)/(-1.0 + ufl.exp(-13.0/2.0 - V/10.0))
        beta_n = 0.125*ufl.exp(-15.0/16.0 - V/80.0)
        F_expressions[2] = (1 - n)*alpha_n - beta_n*n

        # Return results
        return as_vector(F_expressions)

    def num_states(self):
        return 3

    def __str__(self):
        return 'Hodgkin_Huxley_1952 cell model'

    def initial_conditions(self):
        "Return initial conditions for v and s as an Expression."
        return Expression(list(self.default_initial_conditions().keys()), degree=1,
                          **self.default_initial_conditions())
