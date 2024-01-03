#!/usr/bin/env python3

"""
test for error injection
"""

import copy
import os
import math
import pdb
import sys
import time
import tqdm
import numpy as np

from timing_error_cal import calculate_ter


def assign_to_cluster(sign_flip_matrix, sign_flip_order_matrix, output_channel_index, sub_matrix_index,
                      sub_matrix_flip, cluster_limit):
    # pdb.set_trace()
    for cluster_index in sign_flip_order_matrix[:, output_channel_index]:
        flip_count = sign_flip_matrix[cluster_index, output_channel_index]
        if len(sub_matrix_index[cluster_index]) < cluster_limit:
            sub_matrix_index[cluster_index].append(output_channel_index)
            sub_matrix_flip[cluster_index].append(flip_count)
            break
        else:
            if flip_count >= max(
                    sub_matrix_flip[cluster_index]):  # flip > max_flip in current cluster, try another cluster
                continue
            else:  # flip < max_flip in current cluster, pop the boundary point
                max_flip_index = sub_matrix_flip[cluster_index].index(max(sub_matrix_flip[cluster_index]))
                sub_matrix_flip[cluster_index].pop(max_flip_index)
                output_channel_index_reassign = sub_matrix_index[cluster_index].pop(
                    max_flip_index)  # reassign the boundary point
                sub_matrix_index[cluster_index].append(output_channel_index)  # add to current cluster
                sub_matrix_flip[cluster_index].append(flip_count)
                assign_to_cluster(sign_flip_matrix, sign_flip_order_matrix, output_channel_index_reassign,
                                  sub_matrix_index, sub_matrix_flip, cluster_limit)
                break
    return 0


def channel_sign_flip_statistic(channel, sequence):
    channel = channel[sequence]
    sign_flip_count = 0
    for i, weight_i in enumerate(channel):
        if i == 0:
            previous_weight = 0
        else:
            if np.sign(weight_i) * np.sign(previous_weight) == -1:
                sign_flip_count += 1
            previous_weight = weight_i
    return sign_flip_count


def weight_matrix_sign_flip_statistic(weight_matrix):
    sign_compare_matrix = np.sign(weight_matrix[1:, :]) * np.sign(weight_matrix[:-1, :])
    sign_flip_channel = np.sum(sign_compare_matrix == -1, axis=0)
    return sign_flip_channel


if __name__ == "__main__":
    weight_matrix_test = np.int16([[1, 2, 3], [-1, 2, 3], [2, -3, 4], [-1, 2, 3]])

    sign_flip_matrix = np.int16([[10, 9, 10], [12, 13, 11], [21, 22, 23]])
    sign_flip_order_matrix = np.argsort(sign_flip_matrix, axis=0)

    sub_matrix_index = [[], [], []]
    sub_matrix_flip = [[], [], []]
    cluster_limit = 1
    calculate_ter(dta_report_file="/home/zzd/zzd_data/CLionProjects/dta/reports/all_report.dat",
                              ter_report_file='./log/1.csv',
                              clock_cycle=496, total_cycle=40255488, filter_size=2304)
    # assign_to_cluster(sign_flip_matrix, sign_flip_order_matrix, 0, sub_matrix_index, sub_matrix_flip, cluster_limit)
