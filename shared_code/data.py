from fenics import *
from fenics_adjoint import *
import numpy as np
import numpy.random
import pickle

import torch


class Dataset(object):
    def __init__(self, filename, t0, t1, dt, precision=2, obs_func=None):
        self.file = XDMFFile(filename)
        self.filename = filename
        n_obs = round((t1 - t0)/dt) + 1
        self.time_points = [round(t0 + i*dt, precision) for i in range(n_obs)]
        self.precision = precision

        # Only used for writing
        self.current_write_idx = 0

        # Used for interpolation between different function spaces (read only).
        self.obs_func = obs_func

        self.has_written = False

    def read(self, func, t, name="data"):
        t = round(t, self.precision)
        if t in self.time_points:
            idx = self.time_points.index(t)
        else:
            print("Time {} not found in dataset.".format(t))
            raise ValueError
        if self.obs_func is not None:
            self.file.read_checkpoint(self.obs_func, name, idx)
            func = interpolate(self.obs_func, func.function_space())
        else:
            self.file.read_checkpoint(func, name, idx)
        return func

    def write(self, func, t, name="data"):
        t = round(t, self.precision)
        if t not in self.time_points:
            raise ValueError("Time {} not found in dataset.".format(t))
        idx = self.time_points.index(t)
        append = self.has_written
        self.file.write_checkpoint(func, name, t, XDMFFile.Encoding.HDF5, append)
        self.has_written = True

    def reload_file(self):
        if hasattr(self, "file") and self.file is not None:
            return False
        self.file = XDMFFile(self.filename)
        return True

    def __del__(self):
        if hasattr(self, "file") and self.file is not None:
            self.file.close()

    def __getstate__(self):
        r = self.__dict__.copy()
        r["file"] = None
        r["obs_func"] = None
        return r


class Sampler(object):
    def __init__(self, dataset, t0, t1, dt, obs_func):
        self.dataset = dataset
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.obs_func = obs_func
        n_obs = round((t1 - t0) / dt)
        self.obs_times = [round(t0 + dt * i, dataset.precision) for i in range(n_obs + 1)]

    def initial_condition(self):
        return self.dataset.read(self.obs_func, self.t0)

    def get_observation(self, t):
        t = round(t, self.dataset.precision)
        if t in self.obs_times:
            return self.dataset.read(self.obs_func, t)
        return None

    def reload_state(self, obs_func):
        self.dataset.reload_file()
        self.obs_func = obs_func

    def __getstate__(self):
        r = self.__dict__.copy()
        r["obs_func"] = None
        return r

    def save(self, filename):
        with open(filename, "wb") as f:
            return pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def auto_generate_coordinates(self):
        """Returns coordinates
        """
        coords = self.obs_func.function_space().tabulate_dof_coordinates()
        r = []
        for t in self.obs_times:
            ts = np.expand_dims(np.full(coords.shape[:-1], t), -1)
            r.append(np.concatenate((ts, coords), axis=-1))

        return np.stack(r)

    def to_torch(self, coordinates=None):
        """Returns a [time_steps, sample_points, ...] shaped torch tensor + coordinates [t, x].
        """
        if coordinates is None:
            coordinates = self.auto_generate_coordinates()

        r = []
        for coords in coordinates:
            t = coords.item(0)
            obs = self.get_observation(t)
            values = get_sample_points(obs, coords[..., 1:])
            r.append(values)

        return torch.tensor(r, dtype=torch.float64), coordinates


class Subsampler(Sampler):
    def __init__(self, dataset, t0, t1, dt, obs_func, keep=0.5):
        super().__init__(dataset, t0, t1, dt, obs_func)
        sub_obs_times = [t0, t1] + list(numpy.random.choice(self.obs_times[1:-1],
                                                            round(keep * len(self.obs_times)) - 2,
                                                            replace=False))
        self.obs_times = sub_obs_times
        print("Subset time points chosen: {}".format(self.obs_times))


class TestSampler(Sampler):
    def __init__(self, dataset, t0, t1, dt, obs_func, subsampler):
        super().__init__(dataset, t0, t1, dt, obs_func)
        self.subsampler = subsampler

    def get_observation(self, t):
        obs = self.subsampler.get_observation(t)
        if obs is None:
            return super().get_observation(t)
        return None


class DeterministicSubsampler(Sampler):
    def __init__(self, dataset, t0, t1, dt, obs_func, keep=0.5):
        super().__init__(dataset, t0, t1, dt, obs_func)

        sub_obs_indices = np.linspace(0, len(self.obs_times) - 1, round(keep * len(self.obs_times)), dtype=int)
        sub_obs_times = [self.obs_times[i] for i in sub_obs_indices]
        self.obs_times = sub_obs_times


class NoisySubsampler(DeterministicSubsampler):
    def __init__(self, dataset, t0, t1, dt, obs_func, keep=0.5, noise_distribution="flat", noise_level=0.1):
        super().__init__(dataset, t0, t1, dt, obs_func, keep)
        self.noise_level = noise_level
        self.noise_distribution = noise_distribution
        self.noise_random_state = None
        self.create_noise()

    def create_noise(self):
        self.noise_random_state = numpy.random.get_state()
        noise_distribution = self.noise_distribution
        noise_level = self.noise_level
        obs_func = self.obs_func
        if noise_distribution == "flat":
            noise = obs_func.copy(deepcopy=True)
            var = 0.
            for t in self.obs_times:
                obs = super().get_observation(t)
                max_value = max(-obs.vector().min(), obs.vector().max())
                if var < max_value:
                    var = max_value
            noise.vector()[:] = noise_level * var * numpy.random.randn(noise.function_space().dim())
            self.noise = noise
        elif noise_distribution == "flat(t)":
            self.noise = []
            for t in self.obs_times:
                noise = obs_func.copy(deepcopy=True)
                obs = super().get_observation(t)
                var = obs.vector().max() - obs.vector().min()
                noise.vector()[:] = noise_level * var * numpy.random.randn(noise.function_space().dim())
                self.noise.append(noise)
        elif noise_distribution == "snr":
            self.noise = []
            signals = []
            for t in self.obs_times:
                obs = super().get_observation(t)
                signals.append(obs.vector().max() - obs.vector().min())
            for signal in signals:
                sigma = signal / noise_level
                noise = obs_func.copy(deepcopy=True)
                noise.vector()[:] = sigma * numpy.random.randn(noise.function_space().dim())
                self.noise.append(noise)
        elif noise_distribution == "snr_L2":
            self.noise = []
            signals = []
            for t in self.obs_times:
                obs = super().get_observation(t)
                signals.append(sqrt(assemble(obs ** 2 * dx)))
            for i, signal in enumerate(signals):
                sigma = signal / noise_level
                noise = obs_func.copy(deepcopy=True)
                noise.vector()[:] = sigma * numpy.random.randn(noise.function_space().dim())
                self.noise.append(noise)

    def noisy_observation(self, obs, t):
        if obs is None:
            return obs

        t = round(t, self.dataset.precision)
        if self.noise_distribution == "flat":
            obs.vector()[:] += self.noise.vector()
        elif self.noise_distribution in ("flat(t)", "snr", "snr_L2"):
            obs.vector()[:] += self.noise[self.obs_times.index(t)].vector()
        return obs

    def initial_condition(self):
        return super().initial_condition()

    def get_observation(self, t):
        return self.noisy_observation(super().get_observation(t), t)

    def reload_state(self, obs_func):
        super().reload_state(obs_func)

        # Reproduce noise
        tmp_state = numpy.random.get_state()
        numpy.random.set_state(self.noise_random_state)
        self.create_noise()
        numpy.random.set_state(tmp_state)

    def __getstate__(self):
        r = super().__getstate__()
        r["noise"] = None
        return r


def get_sample_points(func, coords):
    """

    func: fenics function to sample.
    coords: the coordinates at which to sample.

    """
    values = []
    for x in coords:
        v = func(x)
        if isinstance(v, float):
            v = [v]
        values.append(v)
    return values

