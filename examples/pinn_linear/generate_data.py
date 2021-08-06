from fenics import *
from fenics_adjoint import *

from model import DataModel
from functional import DataWriter

from shared_code.data import Dataset


model = DataModel(Nx=4096)
t0, t1 = (0., 0.1)
dt = 0.0001
precision = 4

dataset = Dataset("data/data.xdmf", t0, t1, dt, precision)
context = DataWriter(dataset)

x, *_ = SpatialCoordinate(model.mesh)
ic = project(model.initial_condition(x), model.function_space)
context.solver_step(ic, 0.)

model.forward(ic, 0., 0.1, 0.0001, model.ground_truth, context)
