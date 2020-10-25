import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


def one_hot_encode(actions, num_actions, device):
    one_hot_actions = torch.zeros(len(actions), num_actions, dtype=torch.float, device=device)
    one_hot_actions[torch.arange(len(actions)), actions] = 1
    return one_hot_actions


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden1_dim=256, hidden2_dim=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Actor(DQN):

    def __init__(self, input_dim, output_dim, hidden1_dim=256, hidden2_dim=512):
        super(Actor, self).__init__(input_dim, output_dim, hidden1_dim, hidden2_dim)

    def forward(self, x):
        return torch.tanh(super().forward(x))


class Critic(DQN):

    def __init__(self, input_dim, output_dim, action_dim, hidden1_dim=256, hidden2_dim=512):
        super(Critic, self).__init__(input_dim, output_dim, hidden1_dim, hidden2_dim)
        self.fc2 = nn.Linear(hidden1_dim + action_dim, hidden2_dim)

    def forward(self, x, a=None):
        assert a is not None
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(torch.cat((x, a), dim=-1)))
        return self.fc3(x)


class PolicyBase:
    def __init__(self, state_dim, action_dim, device, gamma=0.99, latent=False):
        self.device = device
        self.num_states = state_dim
        self.num_actions = action_dim
        self.gamma = gamma
        self.latent = latent

    def __repr__(self):
        return "Base"

    def select_action(self, state, eps=None):
        raise NotImplementedError

    def optimize(self, memory, Transition, rms=None):
        pass

    def update_target_policy(self):
        pass


class PolicyDQN(PolicyBase):

    def __init__(self, state_dim, action_dim, device,
                 lr=5e-4, batch_size=128, gamma=0.99,
                 eps_start=1., eps_decay=200, eps_end=0.05,
                 target_update=10, func_encode_action=one_hot_encode,
                 latent=False, double=False):
        # conf
        super(PolicyDQN, self).__init__(state_dim, action_dim, device, gamma=gamma, latent=latent)
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.i_step = 0
        self.i_episode = 0
        self.target_update = target_update
        self.func_encode_action = func_encode_action
        self.double = double

        # model
        self.policy_net = DQN(self.num_states, self.num_actions).to(self.device)
        self.target_net = DQN(self.num_states, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def __repr__(self):
        return "DQN"

    def select_action(self, state, eps=None):
        """
            Return argmax_a Q(s,a)
        """
        if type(state) is not torch.Tensor:
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        assert len(state.size()) == 1  # only allow one state
        if eps is None:
            eps = self.eps
        if np.random.uniform() > eps:
            with torch.no_grad():
                action = self.policy_net(state).argmax().item()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def select_action_in_batch(self, states):
        with torch.no_grad():
            actions = self.policy_net(states).argmax(dim=1)
        assert actions.size(0) == actions.size(0)
        return actions

    def calc_value_in_batch(self, states, actions):
        with torch.no_grad():
            return self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    def optimize(self, memory, Transition, rms=None):
        """
            Optimize DQN
        """
        if len(memory) < self.batch_size:
            return
        transitions, weights, indices = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state)  # [B, D_state]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_state_batch = torch.stack([s for s in batch.next_state if s is not None])  # [B, D_state]
        if rms is not None:
            state_batch = rms.normalize(state_batch)
            non_final_next_state_batch = rms.normalize(non_final_next_state_batch)

        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device)  # [B,]
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)  # [B,]
        dt_batch = torch.tensor(batch.dt, dtype=torch.float, device=self.device)  # [B,]

        if self.latent:
            latent_state_batch = torch.stack(batch.latent_state)  # [B, D_latent]
            state_batch = torch.cat((state_batch, latent_state_batch), dim=-1)  # [B, D_state+D_latent]
            non_final_next_latent_state_batch = torch.stack(
                [s for s in batch.next_latent_state if s is not None])  # [B, D_latent]
            non_final_next_state_batch = torch.cat((non_final_next_state_batch, non_final_next_latent_state_batch),
                                                   dim=-1)  # [B, D_state+D_latent]

        assert state_batch.size(0) == self.batch_size
        assert len(reward_batch.size()) == 1

        # compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)  # [B,]

        # compute max_a Q(s_{t+1}, a) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if self.double:
            next_action_batch = self.policy_net(non_final_next_state_batch).argmax(1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_state_batch) \
                .gather(1, next_action_batch.unsqueeze(1)).squeeze(1)
        else:
            next_state_values[non_final_mask] = self.target_net(non_final_next_state_batch).max(1)[0].detach()

        # compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma ** dt_batch) * next_state_values

        # update priority based on TD error
        errors = torch.abs(expected_state_action_values - state_action_values).tolist()
        memory.priority_update(indices, errors)

        # compute loss
        loss = torch.mean(torch.tensor(weights, dtype=torch.float, device=self.device) *
                          self.criterion(state_action_values, expected_state_action_values))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def calc_td_error(self, state, next_state, action, reward, latent_state, next_latent_state, dt):
        """
            Calculate the TD error |Q(s,a) - R(s,a) - gamma^dt * max_a' Q(s',a')|
            :param (next_)state, [D_state,]
                   (next_)latent_state, [D_latent,]
                   action, int
                   reward, int
                   dt, int
        """
        state_cat = state if not self.latent else torch.cat((state, latent_state))
        state_action_value = self.policy_net(state_cat)[action].detach()
        if next_state is not None:
            next_state_cat = next_state if not self.latent else torch.cat((next_state, next_latent_state))
            expected_state_action_value = reward + (self.gamma ** dt) * self.target_net(next_state_cat).max().detach()
        else:
            expected_state_action_value = reward
        return torch.abs(state_action_value - expected_state_action_value).item()

    @property
    def eps(self):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.i_step / self.eps_decay)
        self.i_step += 1
        return eps

    def update_target_policy(self):
        """
            Hard update
        """
        self.i_episode += 1
        if self.i_episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class PolicyDDPG(PolicyBase):

    def __init__(self, state_dim, action_dim, device,
                 actor_lr=1e-4, critic_lr=1e-3, batch_size=128, gamma=0.99,
                 target_update=0.001, func_encode_action=lambda x: x, latent=False):
        # conf
        super(PolicyDDPG, self).__init__(state_dim, action_dim, device, gamma=gamma, latent=latent)
        self.batch_size = batch_size
        self.target_update = target_update
        self.func_encode_action = lambda x, y, z: torch.tensor(func_encode_action(x),
                                                               dtype=torch.float, device=self.device)

        # model
        self.policy_actor = Actor(self.num_states, self.num_actions, hidden1_dim=64, hidden2_dim=64).to(self.device)
        self.target_actor = Actor(self.num_states, self.num_actions, hidden1_dim=64, hidden2_dim=64).to(self.device)
        self.target_actor.load_state_dict(self.policy_actor.state_dict())
        self.target_actor.eval()
        self.optimizer_actor = optim.Adam(self.policy_actor.parameters(), lr=actor_lr)
        self.policy_critic = Critic(self.num_states, 1, self.num_actions, hidden1_dim=64,
                                    hidden2_dim=64).to(self.device)
        self.target_critic = Critic(self.num_states, 1, self.num_actions, hidden1_dim=64,
                                    hidden2_dim=64).to(self.device)
        self.target_critic.load_state_dict(self.policy_critic.state_dict())
        self.target_critic.eval()
        self.optimizer_critic = optim.Adam(self.policy_critic.parameters(), lr=critic_lr)
        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "DDPG"

    def select_action(self, state, eps=None):
        """
            Return actor(s)
        """
        if type(state) is not torch.Tensor:
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        assert len(state.size()) == 1  # only allow one state
        if eps is not None and np.random.uniform() < eps:
            action = np.random.uniform(-1, 1, size=self.num_actions)
        else:
            with torch.no_grad():
                action = self.policy_actor(state).cpu().numpy()
            action += self.noise
            action = np.clip(action, -1, 1)
        return action

    def select_action_in_batch(self, states, noise=True):
        with torch.no_grad():
            actions = self.policy_actor(states)
        if noise:
            actions += torch.empty_like(actions, dtype=torch.float, device=self.device).normal_(0, 0.1)
        actions = torch.clamp(actions, -1, 1)
        return actions

    def calc_value_in_batch(self, states, actions):
        with torch.no_grad():
            return self.policy_critic(states, actions).squeeze(-1)

    def optimize(self, memory, Transition, rms=None):
        """
            Optimize DDPG
        """
        if len(memory) < self.batch_size:
            return
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state)  # [B, D_state]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_state_batch = torch.stack([s for s in batch.next_state if s is not None])  # [B, D_state]
        if rms is not None:
            state_batch = rms.normalize(state_batch)
            non_final_next_state_batch = rms.normalize(non_final_next_state_batch)

        action_batch = torch.stack([self.func_encode_action(a, self.num_actions, self.device)
                                    for a in batch.action])  # [B, D_action]
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)  # [B,]
        dt_batch = torch.tensor(batch.dt, dtype=torch.float, device=self.device)  # [B,]

        if self.latent:
            latent_state_batch = torch.stack(batch.latent_state)  # [B, D_latent]
            state_batch = torch.cat((state_batch, latent_state_batch), dim=-1)  # [B, D_state+D_latent]
            non_final_next_latent_state_batch = torch.stack(
                [s for s in batch.next_latent_state if s is not None])  # [B, D_latent]
            non_final_next_state_batch = torch.cat((non_final_next_state_batch, non_final_next_latent_state_batch),
                                                   dim=-1)  # [B, D_state+D_latent]

        assert state_batch.size(0) == self.batch_size
        assert len(reward_batch.size()) == 1

        # compute Q(s_t, a)
        state_action_values = self.policy_critic(state_batch, action_batch).squeeze(1)  # [B,]

        # compute max_a Q(s_{t+1}, a) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_action_batch = self.target_actor(non_final_next_state_batch).detach()
        next_state_values[non_final_mask] = self.target_critic(non_final_next_state_batch,
                                                               next_action_batch).squeeze(1).detach()

        # compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma ** dt_batch) * next_state_values

        # compute critic loss
        critic_loss = self.criterion(state_action_values, expected_state_action_values)

        # optimize the critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # compute actor loss
        actor_loss = -self.policy_critic(state_batch, self.policy_actor(state_batch)).mean()

        # optimize the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # update target
        soft_update(self.target_actor, self.policy_actor, self.target_update)
        soft_update(self.target_critic, self.policy_critic, self.target_update)

        return critic_loss.item(), actor_loss.item()

    @property
    def noise(self):
        return np.random.normal(0, 0.1, size=self.num_actions)
