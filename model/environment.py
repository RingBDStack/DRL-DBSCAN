from utils.utils import *
import numpy as np

"""
    Environment function for RL parameter selection.
    Paper: Automating DBSCAN via Reinforcement Learning
    Source: https://anonymous.4open.science/r/DRL-DBSCAN
"""


def get_reward(extract_features, extract_labels, cur_labels, cur_cluster_num, extract_data_num, extract_masks,
               bump_flag, buffer, e):
    beta = 0.00
    lamda = 0.00
    M = 3

    nmi = metrics.normalized_mutual_info_score(extract_labels, cur_labels)
    reward_nmi = metrics.normalized_mutual_info_score(extract_labels[extract_masks],
                                                      cur_labels[extract_masks])

    if bump_flag != [0, 0]:
        # reward = -1 * abs(bump_flag[0]) - 1 * abs(bump_flag[1])
        im_reward = reward_nmi
    elif (cur_cluster_num < 2) or (cur_cluster_num == extract_data_num):
        im_reward = reward_nmi
    else:
        # reward = metrics.calinski_harabasz_score(extract_features, cur_labels) / 100
        im_reward = reward_nmi

    avg_reward = 0
    if len(buffer) >= M:
        for bu in buffer:
            avg_reward += bu[4]
        avg_reward = im_reward - avg_reward/M

    reward = im_reward + beta * avg_reward - lamda * e

    return reward, round(nmi, 4), im_reward


def get_state(extract_features, cur_labels, cur_cluster_num, extract_data_num, cur_p,
              bump_flag, p_bound):
    if bump_flag != [0, 0]:
        if bump_flag[0] == -1:
            global_states = [cur_p[0], 0, p_bound[0][1] - cur_p[0],
                             cur_p[1] / 100, cur_p[1] / 100 - p_bound[1][0] / 100, p_bound[1][1] / 100 - cur_p[1] / 100,
                             cur_cluster_num / extract_data_num]
        elif bump_flag[0] == 1:
            global_states = [cur_p[0], cur_p[0] - p_bound[0][0], 0,
                             cur_p[1] / 100, cur_p[1] / 100 - p_bound[1][0] / 100, p_bound[1][1] / 100 - cur_p[1] / 100,
                             cur_cluster_num / extract_data_num]
        elif bump_flag[1] == -1:
            global_states = [cur_p[0], cur_p[0] - p_bound[0][0], p_bound[0][1] - cur_p[0],
                             cur_p[1] / 100, 0 / 100, p_bound[1][1] / 100 - cur_p[1] / 100,
                             cur_cluster_num / extract_data_num]
        elif bump_flag[1] == 1:
            global_states = [cur_p[0], cur_p[0] - p_bound[0][0], p_bound[0][1] - cur_p[0],
                             cur_p[1] / 100, cur_p[1] / 100 - p_bound[1][0] / 100, 0 / 100,
                             cur_cluster_num / extract_data_num]
    else:
        global_states = [cur_p[0], cur_p[0] - p_bound[0][0], p_bound[0][1] - cur_p[0],
                         cur_p[1] / 100, cur_p[1] / 100 - p_bound[1][0] / 100, p_bound[1][1] / 100 - cur_p[1] / 100,
                         cur_cluster_num / extract_data_num]
    #return np.array(global_states)

    local_states = []
    lables = np.array(cur_labels)
    features = np.array(extract_features)
    for l in set(list(cur_labels)):
        index_l = np.where(lables==l)
        features_l = features[index_l]
        local_states.append(getLocalStateSample(features_l, features))

    return [[global_states], local_states]


def convergence_judgment(action_log):
    new_action_log = action_log.argmax(axis=0)
    if new_action_log == 4:
        return True
    else:
        return False

def getLocalState(X):
    dist = np.zeros((X.shape[0], X.shape[0]))
    #center_features = np.average(X, axis=0)
    #print(center_features)
    avg_dist = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dist[i, j] = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2))
        avg_dist[i] = np.sum(dist[i, :]) / X.shape[0]
    center_node_index = np.argmin(avg_dist)
    avg_center_dist = avg_dist[center_node_index]
    center_node_feature = list(X[center_node_index])
    max_center_dist = max(dist[center_node_index, :])
    cluster_num = X.shape[0]

    return [avg_center_dist] + [max_center_dist] + [cluster_num / 1000] + center_node_feature

def getLocalStateSample(X, features):

    center_features = np.average(X, axis=0)
    cluster_num = X.shape[0]

    center_features_all = np.average(features, axis=0)
    dist = np.sqrt(np.sum((center_features - center_features_all) ** 2))

    return [dist, cluster_num / 100] + list(center_features)
