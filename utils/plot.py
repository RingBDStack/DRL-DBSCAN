import math
import matplotlib.pyplot as plt
import numpy as np


"""
    Effect plot generation function.
    Paper: Automating DBSCAN via Reinforcement Learning
    Source: https://anonymous.4open.science/r/DRL-DBSCAN
"""


def get_parameter_fig(log_save_path, parameter_log, num):

    fig, ax1 = plt.subplots(figsize=(18, 8.5))
    ax2 = ax1.twinx()

    # lines
    x = [i for i in range(len(parameter_log[0]))]
    ax1.plot(x, parameter_log[0], label="$Eps$", marker='o', color='red', linestyle='-',
             ms=12, linewidth=4, alpha=0.8, markeredgecolor='w', markeredgewidth=2)
    ax2.plot(x, parameter_log[1], label="$MinPts$", marker='s', color='steelblue', linestyle=':',
             ms=12, linewidth=4, alpha=0.8, markeredgecolor='w', markeredgewidth=2)
    ax1.scatter(x[-1], parameter_log[0][-1], alpha=0.8, color='yellowgreen', s=600, zorder=100)
    ax2.scatter(x[-1], parameter_log[1][-1], alpha=0.8, color='yellowgreen', s=600, zorder=100)

    # legend information
    ax1.legend(loc=[0.30, 1.00], fontsize=32, ncol=3, frameon=False)
    ax2.legend(loc=[0.50, 1.00], fontsize=32, ncol=3, frameon=False)

    # axis information
    ax1.grid()
    ax1.set_xlabel("Episode", fontsize=32)
    ax1.set_ylabel("DBSCAN parameters $Eps$", fontsize=32)
    ax2.set_ylabel("DBSCAN parameters $MinPts$", fontsize=32)

    ax1.set_xticks(np.arange(0, len(parameter_log[0]), math.ceil(len(parameter_log[0])/10)))
    ax2.set_xticks(np.arange(0, len(parameter_log[1]), math.ceil(len(parameter_log[1])/10)))
    ax1.set_yticks(np.arange(0, round(max(parameter_log[0]) / 5, 5) * 6,
                             round(max(parameter_log[0]) / 5, 5)))
    ax2.set_yticks(np.arange(0, math.ceil(max(parameter_log[1]) / 5) * 5.1 + 1,
                             math.ceil(max(parameter_log[1]) / 5)))
    ax1.tick_params(labelsize=32)
    ax2.tick_params(labelsize=32)

    plt.subplots_adjust(left=0.10, bottom=0.17, right=0.90, top=0.90, hspace=0, wspace=0)
    plt.savefig(log_save_path + '/' + str(num) + '-init' + '.pdf')

    return


def get_nmi_fig(log_save_path, nmi_log, km_nmi, num):

    plt.figure(figsize=(18, 8.5))

    db_nmi_log = nmi_log
    km_nmi_log = [km_nmi for _ in db_nmi_log]

    # lines
    x = [i for i in range(len(db_nmi_log))]
    plt.plot(x, db_nmi_log, label="DBSCAN", marker='o', color='darkorange', linestyle='-',
             ms=12, linewidth=4, alpha=0.8, markeredgecolor='w', markeredgewidth=2)
    plt.plot(x, km_nmi_log, label="K-means", marker='x', color='slateblue', linestyle='-',
             ms=15, linewidth=2, alpha=0.5, markeredgecolor='slateblue', markeredgewidth=2)

    # legend information
    plt.legend(loc=[0.27, 1.00], fontsize=32, ncol=2, frameon=False)

    # axis information
    plt.grid()
    plt.xlabel("Episode", fontsize=32)
    plt.ylabel("NMI", fontsize=32)

    plt.xticks(np.arange(0, len(db_nmi_log), math.ceil(len(db_nmi_log) / 10)), fontsize=32)
    plt.yticks(np.arange(0, 1.2, 0.2), fontsize=32)

    plt.subplots_adjust(left=0.10, bottom=0.17, right=0.90, top=0.90, hspace=0, wspace=0)
    plt.savefig(log_save_path + '/' + str(num) + '-nmi' + '.pdf')

    return



