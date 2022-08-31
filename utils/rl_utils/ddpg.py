import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class RMMPolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(RMMPolicyNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
    def forward(self, x):
        a1 = torch.sigmoid(self.fc1(x))
        x = torch.cat([x,a1],dim=1)
        a2 = torch.tanh(self.fc2(x))
        return torch.cat([a1,a2],dim=1)
 
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return self.fc2(x)


class TwoLayerFC(torch.nn.Module):
    def __init__(
        self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x: x
    ):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)

        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


class DDPG:
    """DDPG algo"""

    def __init__(
        self,
        num_in_actor,
        num_out_actor,
        num_in_critic,
        hidden_dim,
        discrete,
        action_bound,
        sigma,
        actor_lr,
        critic_lr,
        tau,
        gamma,
        device,
        use_rmm=True,
    ):

        out_fn = (lambda x: x) if discrete else (lambda x: torch.tanh(x) * action_bound)

        if use_rmm:
            self.actor = RMMPolicyNet(
                num_in_actor,
                hidden_dim,
                num_out_actor,
            ).to(device)
            self.target_actor = RMMPolicyNet(
                num_in_actor,
                hidden_dim,
                num_out_actor,
            ).to(device)
        else:
            self.actor = TwoLayerFC(
                num_in_actor,
                num_out_actor,
                hidden_dim,
                activation=F.relu,
                out_fn=out_fn,
            ).to(device)
            self.target_actor = TwoLayerFC(
                num_in_actor,
                num_out_actor,
                hidden_dim,
                activation=F.relu,
                out_fn=out_fn,
            ).to(device)

        self.critic = TwoLayerFC(num_in_critic, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(num_in_critic, 1, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma
        self.action_bound = action_bound
        self.tau = tau
        self.action_dim = num_out_actor
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.expand_dims(state,0), dtype=torch.float).to(self.device)
        action = self.actor(state)[0].detach().cpu().numpy()

        action = action + self.sigma * np.random.randn(self.action_dim)
        action[0]=np.clip(action[0],0,1)
        action[1]=np.clip(action[1],-1,1)
        return action
    def save_state_dict(self,name):
        dicts = {
            "critic":self.critic.state_dict(),
            "target_critic":self.target_critic.state_dict(),
            "actor":self.actor.state_dict(),
            "target_actor":self.target_actor.state_dict()
        }
        torch.save(dicts,name)
    def load_state_dict(self,name):
        dicts = torch.load(name)
        self.critic.load_state_dict(dicts["critic"])
        self.target_critic.load_state_dict(dicts["target_critic"])
        self.actor.load_state_dict(dicts["actor"])
        self.target_actor.load_state_dict(dicts["target_actor"])
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.float)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        next_q_values = self.target_critic(
            torch.cat([next_states, self.target_actor(next_states)], dim=1)
        )
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(
            F.mse_loss(
                self.critic(torch.cat([states, actions], dim=1)),
                q_targets,
            )
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(
            self.critic(
                torch.cat([states, self.actor(states)], dim=1)
            )
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        logging.info(f"update DDPG: actor loss {actor_loss.item():.3f}, critic loss {critic_loss.item():.3f},  ")
        self.soft_update(self.actor, self.target_actor)  # soft-update the target policy net
        self.soft_update(self.critic, self.target_critic)  # soft-update the target Q value net
