from fenics import *
import ufl
import torch

from functools import partial


def get_torch_stimulus(stimulus_id, function_space):
    stimulus = STIMULUS_DICT[stimulus_id]
    kwargs = {}
    if len(stimulus) >= 3:
        # Converter function
        kwargs = stimulus[2](function_space)
    return partial(stimulus[1], **kwargs)


def get_fenics_stimulus(stimulus_id):
    stimulus = STIMULUS_DICT[stimulus_id]
    return stimulus[0]


# STIM ID: 0
def upper_left_stim_fenics(I_s, t):
    I_s.vector()[0] = 100
    return ufl.conditional(ufl.lt(t, 3), 0, ufl.conditional(ufl.lt(t, 4), I_s, 0)) \
        + ufl.conditional(ufl.lt(t, 30), 0, ufl.conditional(ufl.lt(t, 31), I_s, 0))


def upper_left_stim_torch(I_s, t):
    if t >= 3 and t < 4:
        I_s[0] += 100
    elif t >= 30 and t < 31:
        I_s[0] += 100
    return I_s


# STIM ID: 1
def center_stim_fenics(I_s, t):
    V = I_s.function_space()
    mesh = V.mesh()

    # Repeat every 25ms
    frequency = 25
    # Start of first stimulus
    start = 5.0
    # Length, how long is stimulus applied
    length = 1.0

    # Threshold function
    threshold = ufl.cos(ufl.pi / frequency * length)
    # Timer function
    timer = ufl.cos(2 * ufl.pi / frequency * (t - start - length/2))

    x, y = SpatialCoordinate(mesh)
    zero = Constant(0.)
    I_s = project(ufl.conditional(ufl.gt(abs(x - 12.5), 5), zero, ufl.conditional(ufl.gt(abs(y - 12.5), 5), zero, 100)), V,
                  form_compiler_parameters={"quadrature_degree": 1, "quadrature_rule": "vertex",
                                            "representation": "quadrature"})
    return ufl.conditional(ufl.lt(timer, threshold), 0, I_s)


def center_stim_torch(I_s, t, stim_idx):
    if t % 25 >= 5 and t % 25 <= 6:
        I_s[stim_idx] += 100
    return I_s


def center_stim_converter(V):
    dof_coords = V.tabulate_dof_coordinates()
    stim_idx = []
    for i, coord in enumerate(dof_coords):
        if abs(coord[0] - 12.5) <= 5 and abs(coord[1] - 12.5) <= 5:
            stim_idx.append(i)
    print(stim_idx)
    return {"stim_idx": stim_idx}


# STIM ID: 2
def full_stim_fenics(I_s, t):
    I_s.vector()[:] = 100
    return ufl.conditional(ufl.lt(t, 3), 0, ufl.conditional(ufl.lt(t, 4), I_s, 0))


def full_stim_torch(I_s, t):
    if t >= 3 and t < 4:
        I_s += 100
    return I_s


# STIM ID: 3
def center_stim3_fenics(I_s, t):
    V = I_s.function_space()
    mesh = V.mesh()

    # Repeat every 30ms
    frequency = 30
    # Start of first stimulus
    start = 3.0
    # Length, how long is stimulus applied
    length = 1.0

    # Threshold function
    threshold = ufl.cos(ufl.pi / frequency * length)
    # Timer function
    timer = ufl.cos(2 * ufl.pi / frequency * (t - start - length/2))

    x, y = SpatialCoordinate(mesh)
    zero = Constant(0.)
    I_s = project(ufl.conditional(ufl.gt((x - 12.5)**2 + (y - 12.5)**2, 25), zero, 50 * exp(-sqrt((x - 12.5)**2 + (y - 12.5)**2))), V,
                  form_compiler_parameters={"quadrature_degree": 1, "quadrature_rule": "vertex",
                                            "representation": "quadrature"})
    return ufl.conditional(ufl.lt(timer, threshold), 0, I_s)


def center_stim3_torch(I_s, t, stim_idx, values):
    if t % 30 >= 3 and t % 30 <= 4:
        I_s[stim_idx] += values
    return I_s


def center_stim3_converter(V):
    dof_coords = V.tabulate_dof_coordinates()
    stim_idx = []
    values = []
    def stim(coord):
        x, y = coord
        return 50 * exp(-sqrt((x - 12.5)**2 + (y - 12.5)**2))
    for i, coord in enumerate(dof_coords):
        if (coord[0] - 12.5)**2 + (coord[1] - 12.5)**2  <= 25:
            stim_idx.append(i)
            values.append(stim(coord))
    values = torch.tensor(values)
    return {"stim_idx": stim_idx, "values": values}


STIMULUS_DICT = [
    # 0: Default stimulus upper left corner.
    (upper_left_stim_fenics, upper_left_stim_torch),
    # 1: Test stimulus in center.
    (center_stim_fenics, center_stim_torch, center_stim_converter),
    # 2: Full stimulus.
    (full_stim_fenics, full_stim_torch),
    # 3: Test stimulus in center.
    (center_stim3_fenics, center_stim3_torch, center_stim3_converter),
]
