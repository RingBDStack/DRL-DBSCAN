import collections
import sys
import os
import warnings
import argparse
from time import localtime, strftime
import torch
from utils.plot import *
from utils.utils import *
from model.model import DrlDbscan

"""
    Training and testing DRL-DBSCAN.
    Paper: Automating DBSCAN via Reinforcement Learning
    Source: https://anonymous.4open.science/r/DRL-DBSCAN
"""

parser = argparse.ArgumentParser()

# Shape-Pathbased.txt, Shape-Compound.txt, Shape-Aggregation.txt, Shape-D31.txt Stream-Sensor.txt
parser.add_argument('--data_path', default='data/Shape-Pathbased.txt', type=str,
                    help="Path of features and labels")
parser.add_argument('--log_path', default='results/test', type=str,
                    help="Path of results")

# Model dependent args
parser.add_argument('--use_cuda', default=False, action='store_true',
                    help="Use cuda")
parser.add_argument('--train_size', default=0.20, type=float,
                    help="Sample size used to get rewards")
parser.add_argument('--episode_num', default=15, type=int,
                    help="The number of episode")   # Pre-training and Maintenance: 50
parser.add_argument('--block_num', default=1, type=int,
                    help="The number of data blcoks")  # Offline: 1, Online: 16
parser.add_argument('--block_size', default=5040, type=int,
                    help="The size of data block")  # Offline: -, Online: 5040
parser.add_argument('--layer_num', default=3, type=int,
                    help="The number of recursive layer")  # Offline: 3, Online: 6
parser.add_argument('--eps_size', default=5, type=int,
                    help="Eps parameter space size")
parser.add_argument('--min_size', default=4, type=int,
                    help="MinPts parameter space size")
parser.add_argument('--reward_factor', default=0.2, type=float,
                    help="The impact factor of reward")

# TD3 args
parser.add_argument('--device', default="cpu", type=str,
                    help='"cuda" if torch.cuda.is_available() else "cpu".')
parser.add_argument('--batch_size', default=16, type=int,
                    help='"Reinforcement learning for sampling batch size')
parser.add_argument('--step_num', default=30, type=int,
                    help="Maximum number of steps per RL game")


if __name__ == '__main__':

    print('\n+-------------------------------------------------------+\n'
          '* Training and testing DRL-DBSCAN *\n'
          '* Paper: Automating DBSCAN via Reinforcement Learning *\n'
          '* Source: https://anonymous.4open.science/r/DRL-DBSCAN *\n'
          '\n+-------------------------------------------------------+\n'
          )
    # load hyper-parameters
    args = parser.parse_args()

    # generate log folder
    time_log = '/log_' + strftime("%m%d%H%M%S", localtime())
    log_save_path = args.log_path + time_log
    os.mkdir(log_save_path)
    print("Log save path:  ", log_save_path, flush=True)

    # standardize output records and ignore warnings
    warnings.filterwarnings('ignore')
    std = open(log_save_path + '/std.log', 'a')
    sys.stdout = std
    sys.stderr = std

    # CUDA
    use_cuda = args.use_cuda and torch.cuda.is_available()
    print("Using CUDA:  " + str(use_cuda), flush=True)
    print("Running on:  " + str(args.data_path), flush=True)

    # load idx_reward, features and labels
    if "Shape" in args.data_path:
        idx_reward, features, labels = load_data_shape(args.data_path, args.train_size)
        idx_reward, features, labels = [idx_reward], [features], [labels]
    elif "Stream" in args.data_path:
        idx_reward, features, labels = load_data_stream(args.data_path, args.train_size,
                                                        args.block_num, args.block_size)

    # generate parameter space
    print("Train size:  " + str(args.train_size), flush=True)
    p_size, p_step, p_center, p_bound = generate_parameter_space(features[0], args.layer_num,
                                                                 args.eps_size, args.min_size,
                                                                 args.data_path)

    # build a multi-layer agent collection
    agents = []
    for l in range(0, args.layer_num):
        drl = DrlDbscan(p_size, p_step[l], p_center, p_bound, args.device, args.batch_size,
                        args.step_num, features[0].shape[1])
        agents.append(drl)

    # training
    for b in range(0, args.block_num):
        # log path
        if not os.path.exists(args.log_path + '/Block' + str(b)):
            os.mkdir(args.log_path + '/Block' + str(b))
        os.mkdir(args.log_path + '/Block' + str(b) + time_log)
        std = open(args.log_path + '/Block' + str(b) + time_log + '/std.log', 'a')
        sys.stdout = std
        sys.stderr = std

        # k-means
        k_nmi = kmeans_metrics(features[b], labels[b])

        final_reward_test = [0, p_center, 0]
        label_dic_test = set()
        for l in range(0, args.layer_num):
            agent = agents[l]
            print("[ Testing Layer {0} ]".format(l), flush=True)
            # update starting point
            print("Resetting the parameter space......", flush=True)
            agent.reset(final_reward_test)
            cur_labels, cur_cluster_num, p_log = agent.detect(features[b], collections.OrderedDict())
            final_reward_test = [0, p_log[-1], 0]
            d_nmi, d_ami, d_ari = dbscan_metrics(labels[b], cur_labels)
            for p in p_log:
                label_dic_test.add(str(p[0]) + str("+") + str(p[1]))
        with open(args.log_path + '/Block' + str(b) + '/0_test.txt', 'a') as f:
            f.write(str(d_nmi) + "," + str(d_ami) + "," + str(d_ari) + "," +
                    str(final_reward_test[1]) + "," + str(cur_cluster_num) + "," + str(len(label_dic_test)) + '\n')

        max_max_reward = [0, p_center, 0]
        max_reward = [0, p_center, 0]
        label_dic = collections.OrderedDict()
        first_meet_num = 0
        for l in range(0, args.layer_num):
            agent = agents[l]
            agent.reset(max_max_reward)
            max_max_reward_logs = [max_max_reward[0]]
            early_stop = False
            his_hash_size = len(label_dic)
            cur_hash_size = len(label_dic)
            for i in range(1, args.episode_num):
                print('\n+---------------------------------------------------------------+\n'
                      '                Block {0}, Layer {1}, Episode {2}                    '
                      '\n+---------------------------------------------------------------+\n'.format(b, l, i)
                      )
                # train
                print(len(label_dic))
                print("[ Training Layer {0} ]".format(l), flush=True)
                print("The size of Label Hash is: {0}".format(len(label_dic)), flush=True)
                p_logs = np.array([[], []])
                nmi_logs = np.array([])
                # update starting point
                print("Resetting the parameter space......", flush=True)
                agent.reset0()
                # train the l-th layer
                print("Training the {0}-th layer agent......".format(l), flush=True)
                cur_labels, cur_cluster_num, p_log, nmi_log, max_reward = agent.train(i, idx_reward[b], features[b],
                                                                                      labels[b], label_dic,
                                                                                      args.reward_factor)
                p_logs = np.hstack((p_logs, np.array(list(zip(*p_log)))))
                nmi_logs = np.hstack((nmi_logs, np.array(nmi_log)))
                d_nmi, d_ami, d_ari = dbscan_metrics(labels[b], cur_labels)

                # log
                with open(args.log_path + '/Block' + str(b) + time_log + '/init_log.txt', 'a') as f:
                    f.write('episode=' + str(i) + ', layer=' + str(l) + ',K-Means NMI=' + str(k_nmi) + '\n')
                    f.write(str(p_logs) + '\n')
                    f.write(str(nmi_logs) + '\n')
                if max_max_reward[0] < max_reward[0]:
                    max_max_reward = list(max_reward)
                    cur_hash_size = len(label_dic)
                max_max_reward_logs.append(max_max_reward[0])

                # test
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', flush=True)
                print("[ Testing Layer {0} ]".format(l), flush=True)
                # update starting point
                print("Resetting the parameter space......", flush=True)
                agent.reset0()
                cur_labels, cur_cluster_num, p_log = agent.detect(features[b], label_dic)
                d_nmi, d_ami, d_ari = dbscan_metrics(labels[b], cur_labels)

                # early stop
                if len(max_max_reward_logs) > 3 and \
                        max_max_reward_logs[-1] == max_max_reward_logs[-2] == max_max_reward_logs[-3] and \
                        max_max_reward_logs[-1] != max_max_reward_logs[0]:
                    break
            first_meet_num += cur_hash_size - his_hash_size
            if cur_hash_size == his_hash_size:
                print("......Early stop at layer {0}......".format(l), flush=True)
                break

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', flush=True)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', flush=True)
        print("Final Results: ", flush=True)
        print("[ Total Hash Size is {0} ]".format(len(label_dic)), flush=True)
        print("[ The best parameter is {0} ]".format(max_max_reward[1]), flush=True)
        print("[ The best parameter appears at {0} ]".format(first_meet_num), flush=True)
        cur_labels = label_dic[str(max_max_reward[1][0]) + str("+") + str(max_max_reward[1][1])]
        cur_cluster_num = len(set(list(cur_labels)))
        print("[ The number of clusters is {0} ]".format(cur_cluster_num), flush=True)
        nmi, ami, ari = dbscan_metrics(labels[b], cur_labels)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', flush=True)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', flush=True)

        with open(args.log_path + '/Block' + str(b) + '/1_nmi.txt', 'a') as f:
            f.write(str(nmi) + '\n')
        with open(args.log_path + '/Block' + str(b) + '/2_ami.txt', 'a') as f:
            f.write(str(ami) + '\n')
        with open(args.log_path + '/Block' + str(b) + '/3_ari.txt', 'a') as f:
            f.write(str(ari) + '\n')
        with open(args.log_path + '/Block' + str(b) + '/4_eps.txt', 'a') as f:
            f.write(str(max_max_reward[1][0]) + '\n')
        with open(args.log_path + '/Block' + str(b) + '/5_min_samples.txt', 'a') as f:
            f.write(str(max_max_reward[1][1]) + '\n')
        with open(args.log_path + '/Block' + str(b) + '/6_cur_cluster_num.txt', 'a') as f:
            f.write(str(cur_cluster_num) + '\n')
        with open(args.log_path + '/Block' + str(b) + '/7_first_num.txt', 'a') as f:
            f.write(str(first_meet_num) + '\n')
        with open(args.log_path + '/Block' + str(b) + '/8_all_num.txt', 'a') as f:
            f.write(str(len(label_dic)) + '\n')

        max_reward_nmi = 0
        max_nmi = 0
        max_nmi_logs = []
        for cur_labels in label_dic.values():
            reward_nmi = metrics.normalized_mutual_info_score(labels[b][idx_reward[b]], cur_labels[idx_reward[b]])
            nmi = metrics.normalized_mutual_info_score(labels[b], cur_labels)
            if reward_nmi > max_reward_nmi:
                max_reward_nmi, max_nmi = reward_nmi, nmi
            max_nmi_logs.append(max_nmi)
        get_nmi_fig(log_save_path, max_nmi_logs, k_nmi, num="max_nmi_logs")
        with open(args.log_path + '/Block' + str(b) + '/max_nmi_logs.txt', 'a') as f:
            f.write(str(max_nmi_logs) + '\n')



