# -*- coding: utf-8 -*-
import json
import numpy as np

def smoothDistance(list):
    #参数
    max_realtive_v = 20
    max_absolute_a = 8
    t = 0.05
    beta = 0.7
    max_threshold_x = max_realtive_v * t + 2 * max_absolute_a * t * t
    max_threshold_x = 0.45
    min_threshold_x = -max_threshold_x

    smooth_x_list = [x-1 for x in list]

    for i in range(1, len(list)):
        x0 = smooth_x_list[i - 1]
        x1 = smooth_x_list[i]
        print(abs(x1 - x0))
        if abs(x1 - x0) >= max_threshold_x:
            # 这里是异常点
            if x1 >= x0:
                x1 = beta * x0 + (1 - beta) * (x0 + 0.1)
            else:
                x1 = beta * x0 + (1 - beta) * (x0 - 0.1)
        else:
            # 正常点
            x1 = beta * x0 + (1 - beta) * x1

        smooth_x_list[i] = x1

    return smooth_x_list

def smoothVelocity(smooth_x_list,time_list):

    beta = 0.7
    max_threshold_v = 0.8
    print('----------smoothVelocity:',len(smooth_x_list), len(time_list))
    smooth_v_list = []

    smooth_v_list.append((smooth_x_list[1] - smooth_x_list[0])/float(float(time_list[1]) - float(time_list[0])))
    for i in range(1, len(smooth_x_list)):
        smooth_v_list.append((smooth_x_list[i] - smooth_x_list[i - 1])/float(float(time_list[i]) - float(time_list[i-1])))

    for i in range(1, len(smooth_v_list)):
        x0 = smooth_v_list[i - 1]
        x1 = smooth_v_list[i]
        print(abs(x1 - x0))
        if abs(x1 - x0) >= max_threshold_v:
            # 这里是异常点
            if x1 >= x0:
                x1 = beta * x0 + (1 - beta) * (x0 + 0.2)
            else:
                x1 = beta * x0 + (1 - beta) * (x0 - 0.2)
        else:
            # 正常点
            x1 = beta * x0 + (1 - beta) * x1

        smooth_v_list[i] = x1

    return smooth_v_list


def smoothData(video_list, time_list, x_smooth_scale=19, v_smooth_scale=20, mul=1.15):

    pre_list = video_list
    frames = []
    v_s = []
    x_s = []
    frames.append(0)
    v_s.append(pre_list[0]['vx'])
    x_s.append(pre_list[0]['x'])
    for i in range(1, len(pre_list)):
        # print(list[i])
        pre_object = pre_list[i]

        vx = pre_object['vx']
        x = pre_object['x']

        frames.append(i)
        v_s.append(vx)
        x_s.append(x)

    smooth_x_s = smoothDistance(x_s)
    smooth_v_s = smoothVelocity(smooth_x_s,time_list)

    # smooth_v_s = []
    # v_smooth_scale = 25  # 取奇数
    # for i in range(0, len(pre_list)):
    #     half_window = int(v_smooth_scale / 2)
    #     if i < half_window:
    #         x = x_s[i + v_smooth_scale - 1] - x_s[i]
    #         v = x / (float(time_list[i + v_smooth_scale - 1]) - float(time_list[i]))
    #     elif i > len(pre_list) - half_window - 1:
    #         x = x_s[i] - x_s[i - v_smooth_scale + 1]
    #         v = x / (float(time_list[i]) - float(time_list[i - v_smooth_scale + 1]))
    #     else:
    #         x = x_s[i + half_window] - x_s[i - half_window]
    #         v = x / (float(time_list[i + half_window]) - float(time_list[i - half_window]))
    #     smooth_v_s.append(v)
    #
    # smooth_x_s = []
    # x_smooth_scale = 19  # 取奇数
    # for i in range(0, len(pre_list)):
    #     half_window = int(x_smooth_scale / 2)
    #     if i < half_window:
    #         x = x_s[i:i + x_smooth_scale]
    #     elif i > len(pre_list) - half_window - 1:
    #         x = x_s[i:]
    #     else:
    #         x = x_s[i - half_window:i + half_window + 1]
    #     smooth_x = sum(x) / len(x)
    #     smooth_x_s.append(smooth_x)
    #
    # smooth_x_s = [x / mul for x in smooth_x_s]
    
    smooth_result_list = []
    for i  in range(len(pre_list)):
        dict = {'vx': smooth_v_s[i], 'x': smooth_x_s[i], "fid": i}
        smooth_result_list.append(dict)

    return smooth_result_list


