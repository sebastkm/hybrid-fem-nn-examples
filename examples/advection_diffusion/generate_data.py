from fenics import *
from fenics_adjoint import *

from model import DataModel
from functional import DataWriter
import constants
import time
import argparse

from shared_code.data import Dataset
from shared_code.utils import get_precision


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test-mode", type=int, dest="test_mode", default=0,
                        help="The test mode to use (integer) when generating data. 0 is training set. Default: 0")
parser.add_argument("--name", "-n", type=str, dest="name", default="data",
                    help="Name of dataset file. Default: data")
parser.add_argument("--t1", type=float, dest="t1", default=constants.END_TIME,
                    help="The end time. Default is defined in constants.py.")
args = parser.parse_args()

model = DataModel(Nx=constants.GROUND_TRUTH_NX, test_set=args.test_mode)
t0, t1 = (constants.START_TIME, args.t1)
dt = constants.GT_TIME_STEP
precision = max(get_precision(dt), get_precision(t0))

dataset = Dataset(f"data/{args.name}.xdmf", t0, t1, constants.DATASET_DT, precision)
context = DataWriter(dataset)

x, y = SpatialCoordinate(model.mesh)
ic = project(model.initial_condition(x, y), model.function_space)
context.solver_step(ic, 0.)

model.forward(ic, t0, t1, dt, model.ground_truth, context)
