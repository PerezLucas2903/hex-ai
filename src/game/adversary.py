import numpy as np
import torch
# ---------------------------------
# Interface for adversary agents
# ---------------------------------

class Adversary():
    def __init__(self):
        pass

    def select_action(self, state, valid_actions):
        raise NotImplementedError("Subclasses must implement select_action(state)")
    
    def update_weights(self, new_model_state_dict):
        pass
    

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
    

# --------------------------------
# Neural Network-based adversary agent (Choose between the best n actions)
# --------------------------------

class NNAdversary(Adversary):
    def __init__(self, model,n_best_actions=2, device='cpu'):
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.n_best_actions = n_best_actions

    def select_action(self, state, valid_actions):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().squeeze(0)
        # Mask invalid actions
        q_values_invalid_masked = np.full_like(q_values, -np.inf)
        q_values_invalid_masked[valid_actions] = q_values[valid_actions]
        q_values = q_values_invalid_masked
        
        # Get the indices of the n best actions
        num_best_actions = min(self.n_best_actions, len(valid_actions))
        best_actions_indices = np.argsort(q_values)[-num_best_actions:]

        # Choose one of the best actions
        return np.random.choice(best_actions_indices)

    def update_weights(self, new_model_state_dict):
        self.model.load_state_dict(new_model_state_dict)
        self.model.eval()