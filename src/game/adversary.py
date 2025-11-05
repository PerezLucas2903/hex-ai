import numpy as np

# ---------------------------------
# Interface for adversary agents
# ---------------------------------

class Adversary():
    def __init__(self):
        pass

    def select_action(self, state, valid_actions):
        raise NotImplementedError("Subclasses must implement select_action(state)")
    

# --------------------------------
# Random adversary agent
# --------------------------------

class RandomAdversary(Adversary):
    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

    def select_action(self, state, valid_actions):
        return np.random.choice(valid_actions)