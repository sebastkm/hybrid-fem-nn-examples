import uuid
import pickle
from os import path


class Experiment(object):
    pkl_path = "experiments/{}.pkl"

    def __new__(cls, id=None):
        if id is None:
            # Generate a new experiment.
            r = object.__new__(cls)
            r.create_new_experiment()
            return r
        else:
            # Load previous experiment
            return cls.load_experiment(id)

    def __init__(self, *args, **kwargs):
        if hasattr(self, "id"):
            return
        self.id = None
        self.version = "1.1"
        self.args = None
        self.optimization_iterations = 0
        self.optimization_iteration_loss = []
        self.random_state = None
        self.NN = None
        self.parent = None
        self.metrics = None
        self.original_loss = None
        self.noise_level = None
        self.noise_distribution = None

    def create_new_experiment(self):
        self.id = str(uuid.uuid4())
        self.version = "1.1"
        self.args = None
        self.optimization_iterations = 0
        self.optimization_iteration_loss = []
        self.random_state = None
        self.NN = None
        self.parent = None
        self.metrics = None
        self.original_loss = None
        self.noise_level = None
        self.noise_distribution = None

    @classmethod
    def load_experiment(cls, id):
        with open(cls.pkl_path.format(id), "rb") as f:
            return pickle.load(f)

    @classmethod
    def exists(cls, id):
        return path.exists(cls.pkl_path.format(id))

    def save(self, prefix=""):
        with open(f"{prefix}{self.pkl_path.format(self.id)}", "wb") as f:
            return pickle.dump(self, f)

    def opt_callback(self, J):
        self.optimization_iterations += 1
        self.optimization_iteration_loss.append(float(J))

    @classmethod
    def spawn_child(cls, id):
        child = cls.load_experiment(id)
        parent_id = child.id
        child.id = str(uuid.uuid4())
        child.parent = parent_id
        return child

