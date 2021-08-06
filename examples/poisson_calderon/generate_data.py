from fenics import *
from fenics_adjoint import *

from model import Model
from functional import DataWriter

from shared_code.data import Dataset


model = Model(Nx=200, order=2)
t0, t1 = (0., 1.0)
dt = 1.0

dataset = Dataset("data/data.xdmf", t0, t1, dt, 1)
context = DataWriter(dataset)

x, *_ = SpatialCoordinate(model.mesh)
u_data = project(model.boundary_condition(), model.function_space)
context.solver_step(u_data, 0.)

model.forward(u_data, model.ground_truth, context)
