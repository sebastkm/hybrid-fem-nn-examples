import torch

from shared_code.data import Dataset as BasicDataset, Sampler as BasicSampler
import constants


class Dataset:
    def __init__(self, name, t0, t1, dt, precision, obs_func=None):
        self.dataset = BasicDataset(name, t0, t1, dt, precision)

        # Format: tuple of (v, s)
        self.obs_func = obs_func

    def write(self, func, t, name):
        self.dataset.write(func, t, name=name)

    def read(self, func, t, name="v"):
        if self.obs_func is not None:
            self.dataset.obs_func = self.obs_func[0 + 1 * (name == "s")]
        return self.dataset.read(func, t, name=name)

    def __getattr__(self, item):
        return getattr(self.dataset, item)


class DataWriter:
    def __init__(self, dataset):
        self.dataset = dataset

    def solver_step(self, v, s, t):
        t = round(t, self.dataset.precision)
        if t in self.dataset.time_points:
            print(f"Writing data at t = {t}")
            self.dataset.write(v, t, "v")
            self.dataset.write(s, t, "s")


class Sampler:
    def __init__(self, dataset, t0, t1, dt, obs_func, s_func=None, device=None):
        self.sampler = BasicSampler(dataset, t0, t1, dt, obs_func)
        self._cache = {}
        self.s_func = s_func
        self.device = device

    def get_observation(self, t):
        t = float(t)
        key = round(t, constants.T_PRECISION)
        if key not in self._cache:
            obs = self.sampler.get_observation(t)
            if obs is not None:
                vec = torch.tensor(obs.vector().get_local(), device=self.device)
            else:
                vec = None
            self._cache[key] = vec
        return self._cache[key]

    def get_ic_obs(self, t0):
        if self.sampler.t0 >= t0 - 3 * self.sampler.dt:
            return None
        ic_obs = [self.get_observation(t0 + i * self.sampler.dt) for i in range(-3, 1)]
        return torch.stack(ic_obs)

