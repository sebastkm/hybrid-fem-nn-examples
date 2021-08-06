from fenics import *
from fenics_adjoint import *

from data_model import DataModel
from data import DataWriter, Dataset

import constants
import argparse
from stimulus import get_fenics_stimulus

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", "-n", type=str, dest="name", default="data",
                    help="Name of dataset file excluding extension. Default: data")
parser.add_argument("--stimulus", type=int, dest="stimulus", default=0,
                    help="Stimulus ID to use. Default 0.")
parser.add_argument("--t1", type=float, dest="t1", default=constants.END_TIME,
                    help="End time (T). Default is set in constants.py.")
parser.add_argument("--crossed", dest="crossed", action="store_true",
                    help="Use mesh with crossed diagonals. Default is False (right diagonals).")
args = parser.parse_args()


model = DataModel(args.crossed)
t0, t1 = (constants.START_TIME, args.t1)
dt = constants.GT_TIME_STEP
precision = constants.T_PRECISION

dataset = Dataset(f"data/{args.name}.xdmf", t0, t1, constants.DATASET_DT, precision)
context = DataWriter(dataset)

stimulus = get_fenics_stimulus(args.stimulus)

set_log_level(LogLevel.WARNING)
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

model.forward(t0, t1, dt, stimulus, context)

