from src.game.env import HEX
from src.models.model import MLP_QNet,Conv_QNet
from src.models.resnet import ResNet_QNet
from src.game.agent import DQNAgentPER
import numpy as np
import torch
from src.game.adversary import RandomAdversary

if __name__ == "__main__":
    grid_size = 5
    adversary = RandomAdversary()
    env = HEX(grid_size=grid_size, render_mode="plot", adversary=adversary)

    obs_space = env.observation_space 
    
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    q_net = ResNet_QNet(input_shape = (2,grid_size, grid_size), n_actions=n_actions).to(device)
    target_net = ResNet_QNet(input_shape = (2,grid_size, grid_size), n_actions=n_actions).to(device)

    print(q_net)

    path = "models/resnet_qnet_hex.pth"

    agent = DQNAgentPER(
        env=env,
        q_net=q_net,
        target_net=target_net,
        buffer_size=10000,
        batch_size=256,
        gamma=1,
        lr=1e-5,
        sync_every=1000,
        epsilon_start=1,
        epsilon_final=0.01,
        epsilon_decay=15000,
        update_every=grid_size,
        tau=1.0,
        grad_clip=10.0,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=50000,
        seed=42,
        device=device
    )


    try:
        print(f"Loaded model weights from {path}")
        agent.load(path)
    except:
        print(f"No pre-trained model found at {path}. Training from scratch.")

    
    for _ in range(1000):
        returns = agent.train(num_episodes=int(2e3), max_steps_per_episode=grid_size*grid_size, log_every=100, render=False)
        print("Done. Last returns:", returns[-5:])

        agent.save(path)
        agent.save(path[:-4] + f'_{grid_size}.pth')
        print(f"Saved trained model weights to {path}")

        agent.evaluate(num_episodes=100, max_steps_per_episode=grid_size*grid_size, render=False, temperature=0.0)