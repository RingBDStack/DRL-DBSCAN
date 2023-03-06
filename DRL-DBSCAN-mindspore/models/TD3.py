import copy
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

"""
    TD3 algorithm.
    Paper: Automating DBSCAN via Reinforcement Learning
    Source: https://anonymous.4open.science/r/DRL-DBSCAN
"""


class Actor(nn.Cell):
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

        self.W = nn.Dense(global_state_dim, 32, weight_init="uniform")
        self.U = nn.Dense(local_state_dim, 32, weight_init="uniform")
        self.att = nn.Dense(64, 1, weight_init="uniform")

        self.l1 = nn.Dense(64, 256, weight_init="uniform")
        self.l2 = nn.Dense(256, 256, weight_init="uniform")
        self.l3 = nn.Dense(256, action_dim, weight_init="uniform")

        self.max_action = max_action
        self.device = device

    def construct(self, global_states, local_states):
        """
        forward function of the Actor
        :param global_states: global states of the environment
        :param local_states: local states of the environment
        :return: max_action * attention coefficient
        """
        # att
        zeros = ops.Zeros()
        states = zeros((global_states.shape[0], 64), mindspore.float32)
        for i, global_state, local_state in zip(range(global_states.shape[0]), global_states, local_states):
            w_global_state = self.W(global_state)
            u_local_state = self.U(mindspore.Tensor(local_state))

            # torch.Size([u_local_state.shape[0]+1, w_global_state.shape[1]/32)
            broad_cast_to = ops.BroadcastTo((u_local_state.shape[0] + 1, w_global_state.shape[1]))
            global_ = broad_cast_to(w_global_state)
            # torch.Size([u_local_state.shape[0]+1, u_local_state.shape[1]/32)
            concat_op = ops.Concat()

            local_ = concat_op((w_global_state, u_local_state))

            # torch.Size([u_local_state.shape[0]+1)
            cast_op = ops.Cast()
            concat_op2 = ops.Concat(1)
            leaky_relu = nn.LeakyReLU()
            global_local_score = leaky_relu(self.att(concat_op2((global_, local_))))
            reduce_sum = ops.ReduceSum()
            global_local_score_ = ops.Div()(global_local_score, reduce_sum(global_local_score))
            mul = ops.Mul()
            concat_op3 = ops.Concat(1)
            reduce_sum2 = ops.ReduceSum()
            mul_output1 = mul(global_local_score_[0], w_global_state)
            sum = reduce_sum2(mul(global_local_score_[1:], u_local_state), 0).reshape(1, -1)
            relu = ops.ReLU()
            relu_input = concat_op3((mul_output1, sum))
            relu_output = relu(relu_input)
            states[i] = relu_output.flatten()
        a = ops.ReLU()(self.l1(states))
        a = ops.ReLU()(self.l2(a))
        a = ops.sigmoid(self.l3(a))
        return self.max_action * a
        # return 0


class Critic(nn.Cell):
    def __init__(self, global_state_dim, local_state_dim, action_dim, device):
        """
        The structure of critic
        :param global_state_dim: dimension of the global state
        :param local_state_dim: dimension of the local state
        :param action_dim: dimension of the action
        :param device: cuda
        """
        super(Critic, self).__init__()
        self.W = nn.Dense(global_state_dim, 32, weight_init="uniform")
        self.U = nn.Dense(local_state_dim, 32, weight_init="uniform")
        self.att = nn.Dense(64, 1, weight_init="uniform")

        # Q1 architecture
        self.l1 = nn.Dense(64 + action_dim, 256, weight_init="uniform")
        self.l2 = nn.Dense(256, 256, weight_init="uniform")
        self.l3 = nn.Dense(256, 1, weight_init="uniform")

        # Q2 architecture
        self.l4 = nn.Dense(64 + action_dim, 256, weight_init="uniform")
        self.l5 = nn.Dense(256, 256, weight_init="uniform")
        self.l6 = nn.Dense(256, 1, weight_init="uniform")

        self.device = device

    def construct(self, global_states, local_states, actions):
        """
        forward function of the Critic
        :param global_states: global states of the environment
        :param local_states: local states of the environment
        :param actions
        :return: actions * attention coefficient
        """
        zeros = ops.Zeros()
        states = zeros((global_states.shape[0], 64), mindspore.float32)
        print(global_states)
        for i, global_state, local_state in zip(range(global_states.shape[0]), global_states, local_states):
            w_global_state = self.W(global_state)
            u_local_state = self.U(mindspore.Tensor(local_state))

            # torch.Size([u_local_state.shape[0]+1, w_global_state.shape[1]/32)
            broad_cast_to = ops.BroadcastTo((u_local_state.shape[0] + 1, w_global_state.shape[1]))
            global_ = broad_cast_to(w_global_state)
            # torch.Size([u_local_state.shape[0]+1, u_local_state.shape[1]/32)
            concat_op = ops.Concat()

            local_ = concat_op((w_global_state, u_local_state))

            # torch.Size([u_local_state.shape[0]+1)
            cast_op = ops.Cast()
            concat_op2 = ops.Concat(1)
            leaky_relu = nn.LeakyReLU()
            global_local_score = leaky_relu(self.att(concat_op2((global_, local_))))
            reduce_sum = ops.ReduceSum()
            global_local_score_ = ops.Div()(global_local_score, reduce_sum(global_local_score))
            mul = ops.Mul()
            concat_op3 = ops.Concat(1)
            reduce_sum2 = ops.ReduceSum()
            mul_output1 = mul(global_local_score_[0], w_global_state)
            sum = reduce_sum2(mul(global_local_score_[1:], u_local_state), 0).reshape(1, -1)
            relu = ops.ReLU()
            relu_input = concat_op3((mul_output1, sum))
            relu_output = relu(relu_input)
            states[i] = relu_output.flatten()
        concat_op4 = ops.Concat(1)
        actions = actions.astype(mindspore.float32)
        sa = concat_op4((states, actions))

        q1 = ops.ReLU()(self.l1(sa))
        q1 = ops.ReLU()(self.l2(q1))
        q1 = self.l3(q1)

        q2 = ops.ReLU()(self.l4(sa))
        q2 = ops.ReLU()(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, global_states, local_states, actions):
        # att
        zeros = ops.Zeros()
        states = zeros((global_states.shape[0], 64), mindspore.float32)
        for i, global_state, local_state in zip(range(global_states.shape[0]), global_states, local_states):
            w_global_state = self.W(global_state)
            u_local_state = self.U(mindspore.Tensor(local_state))

            # torch.Size([u_local_state.shape[0]+1, w_global_state.shape[1]/32)
            broad_cast_to = ops.BroadcastTo((u_local_state.shape[0] + 1, w_global_state.shape[1]))
            global_ = broad_cast_to(w_global_state)
            # torch.Size([u_local_state.shape[0]+1, u_local_state.shape[1]/32)
            concat_op = ops.Concat()

            local_ = concat_op((w_global_state, u_local_state))

            # torch.Size([u_local_state.shape[0]+1)
            cast_op = ops.Cast()
            concat_op2 = ops.Concat(1)
            leaky_relu = nn.LeakyReLU()
            global_local_score = leaky_relu(self.att(concat_op2((global_, local_))))
            reduce_sum = ops.ReduceSum()
            global_local_score_ = ops.Div()(global_local_score, reduce_sum(global_local_score))
            mul = ops.Mul()
            concat_op3 = ops.Concat(1)
            reduce_sum2 = ops.ReduceSum()
            mul_output1 = mul(global_local_score_[0], w_global_state)
            sum = reduce_sum2(mul(global_local_score_[1:], u_local_state), 0).reshape(1, -1)
            relu = ops.ReLU()
            relu_input = concat_op3((mul_output1, sum))
            relu_output = relu(relu_input)
            states[i] = relu_output.flatten()
        concat_op4 = ops.Concat(1)
        sa = concat_op4((states, actions))

        q1 = ops.ReLU()(self.l1(sa))
        q1 = ops.ReLU()(self.l2(q1))
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

        self.device = mindspore.get_context("device_target")

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
            mindspore.Tensor([self.global_state[i] for i in ind]),
            [self.local_state[i] for i in ind],
            mindspore.Tensor(self.action[ind]),
            mindspore.Tensor([self.next_global_state[i] for i in ind]),
            [self.next_local_state[i] for i in ind],
            mindspore.Tensor(self.reward[ind]),
            mindspore.Tensor(self.not_done[ind])
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
                           self.action_dim, self.max_action, self.device)
        self.actor_target = copy.deepcopy(self.actor)
        actor_parameters = []
        for m in self.actor.parameters_and_names():
            if m[0]:
                actor_parameters.append(m[1])
        self.actor_optimizer = nn.Adam(actor_parameters, learning_rate=3e-4)

        # initialize the Critic and target Critic of the TD3 framework
        self.critic = Critic(self.global_state_dim, self.local_state_dim,
                             self.action_dim, self.device)
        print(self.critic)
        self.critic_target = copy.deepcopy(self.critic)
        critic_parameters = []
        for m in self.critic.parameters_and_names():
            if m[0]:
                critic_parameters.append(m[1])
        self.critic_optimizer = nn.Adam(critic_parameters, learning_rate=3e-4)
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
        global_states_ = mindspore.Tensor(global_states)
        local_states_ = list(local_states)

        test = self.actor(global_states_, local_states_)
        return test.flatten().asnumpy()

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

        min_value = mindspore.Tensor(-self.noise_clip, mindspore.float32)
        max_value = mindspore.Tensor(self.noise_clip, mindspore.float32)
        noise = ops.clip_by_value(ops.standard_normal(action.shape) * self.policy_noise, min_value, max_value)
        # use target actor to choose next action
        next_action = ops.clip_by_value(self.actor_target(next_global_state, next_local_state) + noise, 0, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_global_state, next_local_state, next_action)

        zeros = ops.Zeros()
        list1 = zeros((len(target_Q1), 1), mindspore.float32)
        for i in range(len(target_Q1)):
            if target_Q1[i] > target_Q2[i]:
                list1[i] = target_Q2[i]
            else:
                list1[i] = target_Q1[i]
        target_Q = list1
        target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(global_state, local_state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
                      nn.MSELoss()(current_Q2, target_Q)
        # print("critic_loss")
        # print(critic_loss)


        # Delayed policy updates
        if self.total_iteration % self.policy_freq == 0:

            # Compute actor losse
            mean_op = ops.ReduceMean()
            actor_loss = mean_op(-self.critic.Q1(global_state, local_state, self.actor(global_state, local_state)))

            # print('actor_loss')
            # print(actor_loss)


            # Update the frozen target models
            for param, target_param in zip(self.critic.get_parameters(), self.critic_target.get_parameters()):
                tensor1 = self.tau * param.data + (1 - self.tau) * target_param.data
                print(target_param.data)
                target_param.data.assign_value(tensor1)

            for param, target_param in zip(self.actor.get_parameters(), self.actor_target.get_parameters()):
                tensor2 = self.tau * param.data + (1 - self.tau) * target_param.data
                print(target_param.data)
                target_param.data.assign_value(tensor2)