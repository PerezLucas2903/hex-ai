import gym
from gym import spaces
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
from src.game.adversary import Adversary
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class HEX(gym.Env):
    """
    Simplified HEX environment.
    Two players alternate placing stones (1 or 2) on a grid.
    Player 1 connects top-to-bottom.
    Player 2 connects left-to-right.
    """

    metadata = {"render_modes": ["matrix", "plot"], "render_fps": 30}

    def __init__(self, grid_size : int,
                 adversary : Adversary,
                 render_mode : str = None ,
                 representation_mode : str ='Matrix_Invertion',
                 random_start : bool =False ):
        super().__init__()

        self.grid_size = grid_size
        self.action_space = spaces.Discrete(grid_size * grid_size)
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(grid_size, grid_size),
                                            dtype=np.int8)
        self.adversary = adversary
        self.state = None
        self.steps = 0
        self.turn = 0
        self.max_steps = grid_size * grid_size
        self.render_mode = render_mode
        self.representation_mode = representation_mode

        # Matplotlib figure (for persistent visualization)
        self.fig = None
        self.ax = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.steps = 0
        self.turn = 0
        info = {}

        if np.random.rand() < 0.5:
            # Adversary starts
            self.turn = 1
            valid_actions = self.get_valid_actions()
            adversary_action = self.adversary.select_action(self.get_representation_state(), valid_actions)
            row, col = self.convert_action_by_representation(adversary_action)
            self.state[row, col] = 2  # Adversary is player 2
            self.steps += 1
            self.turn = 0  # Player 1's turn
        
        return self.get_representation_state(), info

    def check_path(self):
        """Check if the current player has connected their respective sides."""
        player = self.turn
        n = self.grid_size
        visited = set()
        queue = deque()

        if player == 0:
            # Player 1 connects top -> bottom
            for c in range(n):
                if self.state[0, c] == 1:
                    queue.append((0, c))
                    visited.add((0, c))
            target_row = n - 1

            while queue:
                r, c = queue.popleft()
                if r == target_row:
                    return 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and self.state[nr, nc] == 1 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        else:
            # Player 2 connects left -> right
            for r in range(n):
                if self.state[r, 0] == 2:
                    queue.append((r, 0))
                    visited.add((r, 0))
            target_col = n - 1

            while queue:
                r, c = queue.popleft()
                if c == target_col:
                    return 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and self.state[nr, nc] == 2 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return -1
    

    def get_representation_state(self):
        state = self.state.copy()
        if self.representation_mode == 'Matrix_Invertion':
            if self.turn == 0:
                return state
            elif self.turn == 1:
                # Invert 1 and 2
                non_zeros = state != 0
                state = -1 * state + 3
                state = state * non_zeros
                
                return state.T
            else:
                raise KeyError("Fez alguma merda no codigo")
        else:
            return state
        
    def convert_action_by_representation(self,action):
        if self.representation_mode == 'Matrix_Invertion':
            if self.turn == 0:
                row = action // self.grid_size
                col = action % self.grid_size
            elif self.turn == 1:
                col = action // self.grid_size
                row = action % self.grid_size
            else:
                raise KeyError("Fez alguma merda no codigo")
        else:
            row = action // self.grid_size
            col = action % self.grid_size

        return row,col


    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Player's move
        row, col = self.convert_action_by_representation(action)
        self.state[row, col] = self.turn + 1
        self.steps += 1
        player_won = self.check_path() == 0

        if player_won:
            reward = 1.0
            terminated = True
            truncated = False
            info = {"winner": self.turn}
            next_state = self.get_representation_state()
            return next_state, reward, terminated, truncated, info
        
        self.turn = (self.turn + 1) % 2

        # Adversary's move
        valid_actions = self.get_valid_actions()
        adversary_action = self.adversary.select_action(self.get_representation_state(), valid_actions)
        row, col = self.convert_action_by_representation(adversary_action)
        self.state[row, col] = self.turn + 1
        self.steps += 1
        adversary_won = self.check_path() == 1

        if adversary_won:
            reward = -1.0
            terminated = True
            truncated = False
            info = {"winner": self.turn}
            self.turn = (self.turn + 1) % 2 # To transpose the board
            next_state = self.get_representation_state()
            return next_state, reward, terminated, truncated, info
        
        self.turn = (self.turn + 1) % 2

        # Continue game
        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps
        info = {"winner": None}
        next_state = self.get_representation_state()
        return next_state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "matrix":
            print(f"\nStep {self.steps}, Turn: Player {self.turn + 1}")
            for row in self.state:
                print(" ".join(["." if x == 0 else ("X" if x == 1 else "O") for x in row]))
            print()

        elif self.render_mode == "plot":
            self._render_plot()

    def _draw_hex(self, x, y, radius=0.5, color="white"):
        """Draw a hexagon centered at (x, y) using plt.plot."""
        angles = np.linspace(0, 2 * np.pi, 7)
        xs = x + radius * np.cos(angles)
        ys = y + radius * np.sin(angles)
        plt.fill(xs, ys, facecolor=color, edgecolor="black", linewidth=1.5)

    def _render_plot(self):
        n = self.grid_size
        if self.fig is None:
            self.fig = plt.figure(figsize=(5, 5))
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            plt.show()

        self.ax.clear()
        self.ax.set_title(f"HEX Board — Step {self.steps} | Player {self.turn + 1}")
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Draw hex grid
        for r in range(n):
            for c in range(n):
                # hexagonal offset layout
                x = c + 0.5 * r
                y = np.sqrt(3) / 2 * r
                if self.state[r, c] == 1:
                    color = "#ff6666"  # red
                elif self.state[r, c] == 2:
                    color = "#66a3ff"  # blue
                else:
                    color = "white"
                self._draw_hex(x, y, radius=0.45, color=color)

        self.ax.set_xlim(-0.5, n + 1)
        self.ax.set_ylim(-1, np.sqrt(3) / 2 * (n + 1))
        plt.pause(0.2)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def get_valid_actions(self):
        if self.turn == 0:
            return [i for i in range(self.grid_size * self.grid_size) if self.state[i // self.grid_size, i % self.grid_size] == 0]
        else:
            return [i for i in range(self.grid_size * self.grid_size) if self.state.T[i // self.grid_size, i % self.grid_size] == 0]


if __name__ == '__main__':
    env = HEX(grid_size=4, render_mode="plot")
    obs, info = env.reset()
    done = False

    print("Starting HEX game simulation...\n")
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print("Game finished!")
            if info.get("winner"):
                print(f"Winner: Player {info['winner']}")
            else:
                print("It's a draw.")
            done = True

    env.close()
