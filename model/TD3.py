import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    TD3 algorithm.
    Paper: Automating DBSCAN via Reinforcement Learning
    Source: https://anonymous.4open.science/r/DRL-DBSCAN
"""


class Actor(nn.Module):
    def __init__(self, global_state_dim, local_state_dim, action_dim, max_action, device):
        """
        The structure of actor
        :param global_state_dim: dimension of the global state
        :param local_state_dim: dimension of the local state
        :param action_dim: dimension of the action
        :param max_action: clip the action step
        :param device: cuda
        """

        super(Actor, self).__init__()

        self.W = nn.Linear(global_state_dim, 32)
        self.U = nn.Linear(local_state_dim, 32)
        self.att = nn.Linear(64, 1)

        self.l1 = nn.Linear(64, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.device = device

    def forward(self, global_states, local_states):
        """
        forward function of the Actor
        :param global_states: global states of the environment
        :param local_states: local states of the environment
        :return: max_action * attention coefficient
        """
        # att
        states = torch.zeros((global_states.shape[0], 64))
        for i, global_state, local_state in zip(range(global_states.shape[0]), global_states, local_states):
            w_global_state = self.W(global_state)
            u_local_state = self.U(torch.FloatTensor(local_state).to(self.device))

            # torch.Size([u_local_state.shape[0]+1, w_global_state.shape[1]/32)
            global_ = w_global_state.expand(u_local_state.shape[0] + 1, w_global_state.shape[1])
            # torch.Size([u_local_state.shape[0]+1, u_local_state.shape[1]/32)
            local_ = torch.cat((w_global_state, u_local_state))

            # torch.Size([u_local_state.shape[0]+1)
            global_local_score = F.leaky_relu(self.att(torch.cat((global_, local_), 1)))
            global_local_score_ = torch.div(global_local_score, torch.sum(global_local_score))

            states[i] = F.relu(torch.cat((torch.mul(global_local_score_[0], w_global_state),
                                          torch.sum(torch.mul(global_local_score_[1:],
                                                              u_local_state), 0).reshape(1,-1)),1))
        a = F.relu(self.l1(states))
        a = F.relu(self.l2(a))
        a = torch.sigmoid(self.l3(a))
        return self.max_action * a


class Critic(nn.Module):
    def __init__(self, global_state_dim, local_state_dim, action_dim, device):
        """
        The structure of critic
        :param global_state_dim: dimension of the global state
        :param local_state_dim: dimension of the local state
        :param action_dim: dimension of the action
        :param device: cuda
        """
        super(Critic, self).__init__()
        self.W = nn.Linear(global_state_dim, 32)
        self.U = nn.Linear(local_state_dim, 32)
        self.att = nn.Linear(64, 1)

        # Q1 architecture
        self.l1 = nn.Linear(64 + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(64 + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.device = device

    def forward(self, global_states, local_states, actions):
        """
        forward function of the Critic
        :param global_states: global states of the environment
        :param local_states: local states of the environment
        :param actions
        :return: actions * attention coefficient
        """
        states = torch.zeros((global_states.shape[0], 64))
        for i, global_state, local_state in zip(range(global_states.shape[0]), global_states, local_states):
            w_global_state = self.W(global_state)
            u_local_state = self.U(torch.FloatTensor(local_state).to(self.device))

            # torch.Size([u_local_state.shape[0]+1, w_global_state.shape[1]/32)
            global_ = w_global_state.expand(u_local_state.shape[0] + 1, w_global_state.shape[1])
            # torch.Size([u_local_state.shape[0]+1, u_local_state.shape[1]/32)
            local_ = torch.cat((w_global_state, u_local_state))

            # torch.Size([u_local_state.shape[0]+1)
            global_local_score = F.leaky_relu(self.att(torch.cat((global_, local_), 1)))
            global_local_score_ = torch.div(global_local_score, torch.sum(global_local_score))

            states[i] = F.relu(torch.cat((torch.mul(global_local_score_[0], w_global_state),
                                          torch.sum(torch.mul(global_local_score_[1:],
                                                              u_local_state), 0).reshape(1, -1)), 1))

        sa = torch.cat((states, actions), 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, global_states, local_states, actions):
        # att
        states = torch.zeros((global_states.shape[0], 64))
        for i, global_state, local_state in zip(range(global_states.shape[0]), global_states, local_states):
            w_global_state = self.W(global_state)
            u_local_state = self.U(torch.FloatTensor(local_state).to(self.device))

            # torch.Size([u_local_state.shape[0]+1, w_global_state.shape[1]/32)
            global_ = w_global_state.expand(u_local_state.shape[0] + 1, w_global_state.shape[1])
            # torch.Size([u_local_state.shape[0]+1, u_local_state.shape[1]/32)
            local_ = torch.cat((w_global_state, u_local_state))

            # torch.Size([u_local_state.shape[0]+1)
            global_local_score = F.leaky_relu(self.att(torch.cat((global_, local_), 1)))
            global_local_score_ = torch.div(global_local_score, torch.sum(global_local_score))

            states[i] = F.relu(torch.cat((torch.mul(global_local_score_[0], w_global_state),
                                          torch.sum(torch.mul(global_local_score_[1:],
                                                              u_local_state), 0).reshape(1, -1)), 1))

        sa = torch.cat((states, actions), 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class ReplayBuffer(object):
    def __init__(self, action_dim, max_size=int(1e6)):
        """
        initialize the replay buffer of TD3
        :param action_dim: dimension of the action
        :param max_size: max size of the action dimension
        """
        self.max_size = max_size
        self.ptr = 0  # pointer
        self.size = 0

        self.global_state = [[] for _ in range(max_size)]
        self.local_state = [[] for _ in range(max_size)]
        self.action = np.zeros((max_size, action_dim))
        self.next_global_state = [[] for _ in range(max_size)]
        self.next_local_state = [[] for _ in range(max_size)]
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cpu")

    def add(self, states, action, next_states, reward, done):
        """
        add the new quintuple to the buffer
        :param states: global state and local state
        :param action: action choosed by the agent
        :param next_states: the new state of the environment after the action has been performed
        :param reward: action reward given by the environment
        :param done: flag of game completion status
        """
        global_state, local_state = states[0], states[1]
        next_global_state, next_local_state = next_states[0], next_states[1]
        self.global_state[self.ptr] = list(global_state)
        self.local_state[self.ptr] = list(local_state)
        self.action[self.ptr] = action
        self.next_global_state[self.ptr] = list(next_global_state)
        self.next_local_state[self.ptr] = list(next_local_state)
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        sample batch size quintuple from the replay buffer
        :param batch_size: number of the quintuple
        :return: sample batch size quintuple
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor([self.global_state[i] for i in ind]).to(self.device),
            [self.local_state[i] for i in ind],
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor([self.next_global_state[i] for i in ind]).to(self.device),
            [self.next_local_state[i] for i in ind],
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class Skylark_TD3():
    def __init__(
            self, global_state_dim, local_state_dim, action_dim, max_action, device_setting,
            gamma=0.1,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2):
        """
        TD3 framework
        :param global_state_dim: dimension of the global state
        :param local_state_dim: dimension of the local state
        :param action_dim: dimension of the action
        :param max_action: clip the action step
        :param device_setting: cuda
        :param gamma:
        :param tau:
        :param policy_noise:
        :param noise_clip: clip the noise
        :param policy_freq:
        """

        # Varies by environment
        self.global_state_dim = global_state_dim
        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device_setting

        # initialize the Actor and target Actor of the TD3 framework
        self.actor = Actor(self.global_state_dim, self.local_state_dim,
                           self.action_dim, self.max_action, self.device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # initialize the Critic and target Critic of the TD3 framework
        self.critic = Critic(self.global_state_dim, self.local_state_dim,
                             self.action_dim, self.device).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.start_timesteps = 1e3  # Time steps initial random policy is used
        self.expl_noise = 2.0  # Std of Gaussian exploration noise

        self.total_iteration = 0

    def select_action(self, states):
        """
        choose action through the Actor
        :param states: global state and local state
        :return: action attention feature
        """

        global_states = [states[0]]
        local_states = [states[1]]
        global_states_ = torch.FloatTensor(global_states).to(self.device)
        local_states_ = list(local_states)

        return self.actor(global_states_, local_states_).cpu().data.numpy().flatten()

    def learn(self, replay_buffer, batch_size=16):
        """
        learning process of the agent
        :param replay_buffer: store the quintuple
        :param batch_size: batch size
        """

        self.total_iteration += 1

        # Sample replay buffer
        global_state, local_state, action, next_global_state, next_local_state, \
        reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            # use target actor to choose next action
            next_action = (
                    self.actor_target(next_global_state, next_local_state) + noise
            ).clamp(0, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_global_state, next_local_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(global_state, local_state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)
        # print("critic_loss")
        # print(critic_loss)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_iteration % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(global_state, local_state, self.actor(global_state, local_state)).mean()

            # print('actor_loss')
            # print(actor_loss)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
