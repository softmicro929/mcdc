# -*- coding: utf-8 -*-
import json
import numpy as np

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

    smooth_v_s = []
    v_smooth_scale = 25  # 取奇数
    for i in range(0, len(pre_list)):
        half_window = int(v_smooth_scale / 2)
        if i < half_window:
            x = x_s[i + v_smooth_scale - 1] - x_s[i]
            v = x / (float(time_list[i + v_smooth_scale - 1]) - float(time_list[i]))
        elif i > len(pre_list) - half_window - 1:
            x = x_s[i] - x_s[i - v_smooth_scale + 1]
            v = x / (float(time_list[i]) - float(time_list[i - v_smooth_scale + 1]))
        else:
            x = x_s[i + half_window] - x_s[i - half_window]
            v = x / (float(time_list[i + half_window]) - float(time_list[i - half_window]))
        smooth_v_s.append(v)

    smooth_x_s = []
    x_smooth_scale = 19  # 取奇数
    for i in range(0, len(pre_list)):
        half_window = int(x_smooth_scale / 2)
        if i < half_window:
            x = x_s[i:i + x_smooth_scale]
        elif i > len(pre_list) - half_window - 1:
            x = x_s[i:]
        else:
            x = x_s[i - half_window:i + half_window + 1]
        smooth_x = sum(x) / len(x)
        smooth_x_s.append(smooth_x)

    smooth_x_s = [x / mul for x in smooth_x_s]
    
    smooth_result_list = []
    for i  in range(len(pre_list)):
        dict = {'vx': smooth_v_s[i], 'x': smooth_x_s[i], "fid": i}
        smooth_result_list.append(dict)

    return smooth_result_list


