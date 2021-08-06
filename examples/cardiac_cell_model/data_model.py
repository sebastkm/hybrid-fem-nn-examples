from fenics import *
import ufl

from shared_code.utils import get_precision

from model import Model
from hodgkin_huxley_1952 import Hodgkin_Huxley_1952

import constants


class DataModel:
    def __init__(self, crossed=False):
        # Create cell model
        self.cellmodel = Hodgkin_Huxley_1952()
        self.num_states = self.cellmodel.num_states()

        # Define mesh and function spaces
        diagonal = "crossed" if crossed else "right"
        self.mesh = RectangleMesh(Point((0, 0)), Point(constants.OMEGA_DIMENSIONS), *constants.FEM_CELLS, diagonal)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.S = VectorFunctionSpace(self.mesh, "CG", 1, dim=self.num_states)

    def forward(self, t0, t1, dt, stimulus, context):
        V, S = self.V, self.S
        num_states = self.num_states
        cellmodel = self.cellmodel

        # Define functions
        v = Function(V)
        s = Function(S)
        s_ = TrialFunction(S)

        s_split = as_vector([s[i] for i in range(0, num_states)])
        v_ = TrialFunction(V)
        w = TestFunction(V)

        test = TestFunction(S)
        q = as_vector([test[i] for i in range(0, num_states)])

        # Define model parameters
        t = t0
        time = Constant(0.)
        dt_ = Constant(dt)  # ms

        I_s = stimulus(Function(V), time)
        chi = 140  # mm^-1
        C_m = 0.01  # uF/mm^2
        M = as_tensor([[0.174 / (chi * C_m), 0], [0, 0.174 / (chi * C_m)]])

        # Define variational forms
        I_ion = Function(V)
        F_v = inner((v_ - v), w)*dx + dt_ * inner(M * grad(v_), grad(w)) * dx - dt_ * I_s * w * dx + dt_ * inner(I_ion, w) * dx
        F_s = inner((s_ - s), q)*dP - dt_ * inner(cellmodel.F(v, s_split), q)*dP

        a_v, L_v = system(F_v)
        a_s, L_s = system(F_s)

        # Set initial conditions

        V_h = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        S_h = VectorElement("CG", self.mesh.ufl_cell(), 1, dim=num_states)
        VS = FunctionSpace(self.mesh, V_h * S_h)
        vs = Function(VS)
        vs.assign(cellmodel.initial_conditions())

        import dolfin
        assigner = dolfin.FunctionAssigner(V, VS.sub(0))
        assigner.assign(v, vs.sub(0))

        assigner = dolfin.FunctionAssigner(S, VS.sub(1))
        assigner.assign(s, vs.sub(1))

        # Save initial v
        context.solver_step(v, s, t)

        precision = max(get_precision(t), get_precision(dt))

        while round(t, precision) < t1:
            t += float(dt)
            time.assign(t)

            # Update cell states
            solve(a_s == L_s, s)

            # https://bitbucket.org/fenics-project/ffc/issues/184/unable-to-use-vertex-quadrature
            I_ion.assign(project(cellmodel.I(v, s_split), V,
                                 form_compiler_parameters={"quadrature_degree": 1, "quadrature_rule": "vertex",
                                                           "representation": "quadrature"}))

            # Update transmembrane potential
            solve(a_v == L_v, v)

            context.solver_step(v, s, t)


def splat(vs, dim):
    if vs.function_space().ufl_element().num_sub_elements() == dim:
        v = vs[0]
        if dim == 2:
            s = vs[1]
        else:
            s = as_vector([vs[i] for i in range(1, dim)])
    else:
        v, s = split(vs)

    return v, s
