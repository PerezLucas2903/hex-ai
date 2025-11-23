from src.game.env import HEX
from src.models.model import MLP_QNet,Conv_QNet
from src.models.resnet import ResNet_QNet
from src.models.attention import Attention_QNet
from src.game.agent import DQNAgentPER
import numpy as np
import torch
from src.game.adversary import RandomAdversary, NNAdversary

if __name__ == "__main__":
    grid_size = 5
    n_attention_layers = 6
    n_dim = 32

    model_adversary = Attention_QNet(n_attention_layers=n_attention_layers, n_dim=n_dim)
    try:
        model_adversary.load_state_dict(torch.load("models/attention_hex_adv.pth")['q_state'])
    except:
        print("No pre-trained adversary model found. Using untrained adversary.")
    adversary = NNAdversary(model_adversary)
    env = HEX(grid_size=grid_size, render_mode="plot", adversary=adversary)

    obs_space = env.observation_space 
    
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    q_net = Attention_QNet(n_attention_layers=n_attention_layers, n_dim=n_dim).to(device)
    target_net = Attention_QNet(n_attention_layers=n_attention_layers, n_dim=n_dim).to(device)

    print(q_net)
    # print model parameters count
    total_params = sum(p.numel() for p in q_net.parameters())
    print(f"Total model parameters: {total_params}")

    path = "models/attention_hex.pth"

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
        epsilon_final=0.1,
        epsilon_decay=15000,
        update_every=grid_size,
        tau=1.0,
        grad_clip=10.0,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=50000,
        device=device
    )


    try:
        print(f"Loaded model weights from {path}")
        agent.load(path)
    except:
        print(f"No pre-trained model found at {path}. Training from scratch.")

    
    for _ in range(1000):
        """
        returns = agent.train(num_episodes=int(2e3), max_steps_per_episode=grid_size*grid_size, log_every=100, render=False)
        print("Done. Last returns:", returns[-5:])

        agent.save(path)
        agent.save(path[:-4] + f'_{grid_size}.pth')
        print(f"Saved trained model weights to {path}")"""

        win_rate = agent.evaluate(num_episodes=100, max_steps_per_episode=grid_size*grid_size, 
                                  render=True)

        if win_rate >= 0.6:
            agent.env.adversary.update_weights(torch.load(path)['q_state'])
            print("Adversary updated.")
            torch.save({'q_state': agent.env.adversary.model.state_dict()}, "models/attention_hex_adv.pth")
            agent.replay.clear()

