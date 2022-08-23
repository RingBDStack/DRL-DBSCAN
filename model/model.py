import numpy as np
from sklearn.cluster import DBSCAN
from model.TD3 import Skylark_TD3, ReplayBuffer
from model.environment import get_reward, get_state, convergence_judgment


"""
    DRL-DBSCAN model.
    Paper: Automating DBSCAN via Reinforcement Learning
    Source: https://anonymous.4open.science/r/DRL-DBSCAN
"""


class DrlDbscan:
    """
        define our deep-reinforcement learning dbscan class

    """
    def __init__(self, p_size, p_step, p_center, p_bound, device, batch_size, step_num, dim):
        """
        initialize the reinforcement learning agent
        :param p_size: parameter space size
        :param p_step: step size
        :param p_center: starting point of the first layer
        :param p_bound: limit bound for the two parameter spaces
        :param device: cuda
        :param batch_size: batch size
        :param step_num: Maximum number of steps per RL game
        :param dim: dimension of feature
        """

        # TD3
        self.agent = Skylark_TD3(global_state_dim=7, local_state_dim=dim+2, action_dim=5,
                                 max_action=1.0, device_setting=device)
        self.replay_buffer = ReplayBuffer(action_dim=5)
        self.batch_size = batch_size
        self.step_num = step_num

        # parameter space: p_center, p_size, p_step, p_bound0
        self.p_center = list(p_center)
        self.p_size, self.p_step, self.p_bound0 = list(p_size), list(p_step), list(p_bound)
        self.p_bound = self.get_parameter_space()

        # cur_p: current parameter
        self.cur_p = list(p_center)
        # logs for score, reward, parameter, action, nmi
        self.score_log, self.reward_log, self.im_reward_log, self.p_log, self.action_log, self.nmi_log = [], [], [], \
                                                                                                         [], [], []

        self.max_reward = [0, list(p_center), 0]

    def reset0(self):
        """
        update the new parameter space and related records
        """
       self.cur_p = list(self.p_center)
       self.score_log, self.reward_log, self.im_reward_log, self.p_log, self.action_log, self.nmi_log = [], [], [], \
                                                                                                        [], [], []

       print("The starting point of the parameter is:  " + str(self.p_center), flush=True)
       print("The parameter space boundary is:  " + str(self.p_bound), flush=True)
       print("The size of the parameter space is:  " + str(self.p_size), flush=True)
       print("The step of the parameter space is:  " + str(self.p_step), flush=True)

   def reset(self, max_reward):
       """
        reset the environment
        :param max_reward: record the max reward
        """

        self.p_center = list(max_reward[1])
        self.p_bound = self.get_parameter_space()
        self.cur_p = list(max_reward[1])
        self.score_log, self.reward_log, self.im_reward_log, self.p_log, self.action_log, self.nmi_log = [], [], [], \
                                                                                                         [], [], []
        self.max_reward = list(max_reward)

        print("The starting point of the parameter is:  " + str(self.p_center), flush=True)
        print("The parameter space boundary is:  " + str(self.p_bound), flush=True)
        print("The size of the parameter space is:  " + str(self.p_size), flush=True)
        print("The step of the parameter space is:  " + str(self.p_step), flush=True)

    def get_parameter_space(self):
        """
        get parameter space of the current layer
        :return: parameter space
        """

        p_bound = [[max(self.p_center[i] - self.p_step[i] * int(self.p_size[i] / 2), self.p_bound0[i][0]),
                    min(self.p_center[i] + self.p_step[i] * int(self.p_size[i] / 2), self.p_bound0[i][1])]
                   for i in range(2)]

        return p_bound

    def action_to_parameters(self, cur_p, action):
        """
        Translate reinforcement learning output actions into specific parameters
        :param cur_p: current parameter
        :param action: current action
        :return: parameters and bump flags
        """

        bump_flag = [0, 0]
        new_p = [0, 0]
        new_action = [0, 0, 0, 0, 0]
        new_action[action.argmax(axis=0)] = 1

        # if the action goes beyond the parameter space, the parameters remain unchanged
        for i in range(2):
            new_p[i] = cur_p[i] - self.p_step[i] * new_action[0 + 2 * i] + \
                       self.p_step[i] * new_action[1 + 2 * i]
            # bump flags for helping judge whether our new parameters out of space
            if new_p[i] < self.p_bound[i][0]:
                new_p[i] = self.p_bound[i][0]
                bump_flag[i] = -1
            elif new_p[i] > self.p_bound[i][1]:
                new_p[i] = self.p_bound[i][1]
                bump_flag[i] = 1
        return new_p, bump_flag

    def stop_processing(self, buffer, final_p, max_p):
        """
        Sample training data and store
        :param buffer: store historical data for training
        :param final_p: reward_factor
        :param max_p: 1 - reward_factor
        """
        buffer.append([self.score_log[-2], self.action_log[-1], self.score_log[-1],
                       self.reward_log[-1], self.im_reward_log[-1]])
        final_reward = buffer[-1][4]
        post_max_reward = buffer[-1][4]
        for bu in reversed(buffer):
            post_max_reward = max(post_max_reward, bu[4])
            bu[3] = final_p * final_reward + max_p * post_max_reward
        # print([bu[3] for bu in buffer])

        for bu in buffer[:-1]:
            self.replay_buffer.add(bu[0], bu[1], bu[2],
                                   bu[3], float(0))
        self.replay_buffer.add(buffer[-1][0], buffer[-1][1], buffer[-1][2],
                               buffer[-1][3], float(1))

        # train
        if self.replay_buffer.size >= self.batch_size:
            for _ in range(len(buffer)):
                self.agent.learn(self.replay_buffer, self.batch_size)

    def train(self, ii, extract_masks, extract_features, extract_labels, label_dic, reward_factor):

        """
        Train DRL-DBSCAN: RL searching for parameters
        :param ii: episode_num
        :param extract_masks: sample serial numbers for rewards
        :param extract_features: features
        :param extract_labels: labels
        :param label_dic: records for parameters and its clustering results (cur_labels)
        :param reward_factor: factors for final reward

        :return: cur_labels, cur_cluster_num, self.p_log, self.nmi_log, self.max_reward
        """

        extract_data_num = extract_features.shape[0]

        # DBSCAN clustering
        if str(self.cur_p[0]) + str("+") + str(self.cur_p[1]) in label_dic:
            cur_labels = label_dic[str(self.cur_p[0]) + str("+") + str(self.cur_p[1])]
        else:
            cur_labels = DBSCAN(eps=self.cur_p[0], min_samples=self.cur_p[1]).fit_predict(extract_features)
            label_dic[str(self.cur_p[0]) + str("+") + str(self.cur_p[1])] = np.array(cur_labels)
        cur_cluster_num = len(set(list(cur_labels)))

        # Get state
        state = get_state(extract_features, cur_labels, cur_cluster_num, extract_data_num,
                          self.cur_p, [0, 0], self.p_bound)

        bump_flag = [0, 0]
        buffer = []

        final_p = reward_factor
        max_p = 1 - reward_factor
        # begin RL game
        for e in range(self.step_num):

            self.score_log.append(state)

            # early stop
            if e >= 2 and convergence_judgment(self.action_log[-1]):
                self.stop_processing(buffer, final_p, max_p)
                print("! Early stop.", flush=True)
                break
            # out of bounds stop
            elif bump_flag != [0, 0]:
                self.stop_processing(buffer, final_p, max_p)
                print("! Out of bounds stop.", flush=True)
                break
            # Timeout stop
            elif e == self.step_num - 1:
                self.stop_processing(buffer, final_p, max_p)
                print("! Timeout stop.", flush=True)
                break
            # play the game
            else:
                if e != 0:
                    buffer.append([self.score_log[-2], self.action_log[-1], self.score_log[-1],
                                   self.reward_log[-1], self.im_reward_log[-1]])

                # train
                # predict actions and obtain parameters based on state
                real_action = self.agent.select_action(self.score_log[-1])
                # print(real_action)
                if ii == 1:
                    new_action = (real_action).clip(0, self.agent.max_action)
                else:
                    new_action = (
                            real_action +
                            np.random.normal(0, self.agent.max_action * self.agent.expl_noise,
                                             size=self.agent.action_dim)
                    ).clip(0, self.agent.max_action)
                new_p, bump_flag = self.action_to_parameters(self.cur_p, new_action)

            if str(new_p[0]) + str("+") + str(new_p[1]) in label_dic:
                cur_labels = label_dic[str(new_p[0]) + str("+") + str(new_p[1])]
            else:
                cur_labels = DBSCAN(eps=new_p[0], min_samples=new_p[1]).fit_predict(extract_features)
                label_dic[str(new_p[0]) + str("+") + str(new_p[1])] = np.array(cur_labels)
            cur_cluster_num = len(set(list(cur_labels)))
            state = get_state(extract_features, cur_labels, cur_cluster_num, extract_data_num,
                              new_p, bump_flag, self.p_bound)

            # get reward
            reward, nmi, im_reward = get_reward(extract_features, extract_labels, cur_labels, cur_cluster_num,
                                                extract_data_num, extract_masks, bump_flag, buffer, e)

            # log
            self.cur_p = list(new_p)
            self.p_log.append(new_p)
            self.action_log.append(new_action)
            self.reward_log.append(reward)
            self.im_reward_log.append(im_reward)
            self.nmi_log.append(nmi)

        # print(self.im_reward_log)
        # print(self.nmi_log)
        # print(self.action_log)
        # print(self.p_log)

        print("! Stop at step {0} with parameter {1}.".format(e, str(self.cur_p)), flush=True)

        if max(self.im_reward_log) > self.max_reward[0]:
            self.max_reward = [max(self.im_reward_log),
                               self.p_log[self.im_reward_log.index(max(self.im_reward_log))],
                               self.nmi_log[self.im_reward_log.index(max(self.im_reward_log))]]
        print("! The current maximum reward {0} appears when the parameter is {1}.".
              format(self.max_reward[2], self.max_reward[1]), flush=True)

        return cur_labels, cur_cluster_num, self.p_log, self.nmi_log, self.max_reward

    def detect(self, extract_features, label_dic):
        """
                            Detect DRL-DBSCAN:
                            :param extract_features: features
                            :param label_dic: records for parameters and its clustering results (cur_labels)

                            :return: cur_labels, cur_cluster_num, p_log
                        """

        extract_data_num = extract_features.shape[0]

        # DBSCAN clustering
        if str(self.cur_p[0]) + str("+") + str(self.cur_p[1]) in label_dic:
            cur_labels = label_dic[str(self.cur_p[0]) + str("+") + str(self.cur_p[1])]
        else:
            cur_labels = DBSCAN(eps=self.cur_p[0], min_samples=self.cur_p[1]).fit_predict(extract_features)
        cur_cluster_num = len(set(list(cur_labels)))

        # Get state
        state = get_state(extract_features, cur_labels, cur_cluster_num, extract_data_num,
                          self.cur_p, [0, 0], self.p_bound)

        # begin RL game
        p_log = []
        action_log = []
        bump_flag = [0, 0]
        cur_p = self.cur_p

        for e in range(self.step_num):
            # early stop
            if e >= 2 and convergence_judgment(action_log[-1]):
                print("! Early stop.", flush=True)
                break
            # out of bounds stop
            elif bump_flag != [0, 0]:
                print("! Out of bounds stop.", flush=True)
                break
            # Timeout stop
            elif e == self.step_num - 1:
                print("! Timeout stop.", flush=True)
                break
            # play the game
            else:
                # predict actions and obtain parameters based on state
                new_action = (self.agent.select_action(state)).clip(0, self.agent.max_action)
                new_p, bump_flag = self.action_to_parameters(cur_p, new_action)

            # get new state
            if str(new_p[0]) + str("+") + str(new_p[1]) in label_dic:
                cur_labels = label_dic[str(new_p[0]) + str("+") + str(new_p[1])]
            else:
                cur_labels = DBSCAN(eps=new_p[0], min_samples=new_p[1]).fit_predict(extract_features)
            cur_cluster_num = len(set(list(cur_labels)))

            state = get_state(extract_features, cur_labels, cur_cluster_num, extract_data_num,
                              new_p, bump_flag, self.p_bound)
            cur_p = list(new_p)
            p_log.append(new_p)
            action_log.append(new_action)
        print(action_log)
        print(p_log)
        print("! Stop at step {0} with parameter {1}.".format(e, str(cur_p)), flush=True)

        return cur_labels, cur_cluster_num, p_log





