from shared_code.utils import get_precision

START_TIME = 0.
END_TIME = 30.0
GT_TIME_STEP = 0.001
T_PRECISION = max(get_precision(START_TIME), get_precision(GT_TIME_STEP))
NUM_ALPHAS = 1
FEM_CELLS = (15, 15)
OMEGA_DIMENSIONS = (25, 25)
DATASET_DT = 0.01
