from fenics import *
from fenics_adjoint import *

import math


class RobustReducedFunctional(ReducedFunctional):
    def __call__(self, *args, **kwargs):
        try:
            value = super().__call__(*args, **kwargs)
        except:
            import traceback
            print(traceback.format_exc())
            print("Warning: Forward computation crashed. Resuming...")
            value = math.nan

        if math.isnan(value) or math.isinf(value):
            self.functional.block_variable.checkpoint = 1e+3
            value = self.scale * self.functional.block_variable.checkpoint
        return value


def train(Jhat,
          method="L-BFGS-B",
          callback=lambda *args: None,
          tol=None,
          **kwargs):

    state = {"backup_values": [],
             "loss": []}
    def opt_callback(*args, **kwargs):
        state["loss"].append(Jhat.scale * Jhat.functional.block_variable.checkpoint)

        r = []
        controls = Jhat.controls
        for ctrl in controls:
            r.append(ctrl.tape_value()._ad_create_checkpoint())
        state["backup_values"] = r

        callback(*args)

    minimize_kwargs = {}
    if tol is not None:
        minimize_kwargs["tol"] = tol

    options = {"disp": True}
    options.update(kwargs)

    if method == "L-BFGS-B" and "maxfun" not in options:
        if "maxiter" in options:
            options["maxfun"] = 10 * options["maxiter"]

    try:
        weights = minimize(Jhat, method=method, callback=opt_callback,
                           options=options, **minimize_kwargs)
    except:
        import traceback
        print(traceback.format_exc())
        weights = state["backup_values"]

    return weights, state["loss"]


