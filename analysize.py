# -*- coding: utf-8 -*-
import json
import numpy as np
import csv
from matplotlib import pyplot as plt


# import MCDC.neton as Neton


def getFrameGap(time_gap_times):
    time_list = []
    time_f = open(time_gap_times)
    while True:
        line = time_f.readline().strip('\n')
        if not line:
            break
        time_list.append(line)
    time_f.close()
    return time_list


def validateVideo():
    filepath = '/Users/wangshuainan/Desktop/valid_video_01_pre.json'
    with open(filepath) as f:
        valid_video = json.load(f)

    list = valid_video['frame_data']
    # 将时间差读进list
    time_list = getFrameGap('/Users/wangshuainan/Desktop/mcdc_data/valid/valid_video_00_time.txt')

    with open('./names.csv', 'w') as csvfile:
        fieldnames = ['No', 'vx', 'x']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        frames = []
        vx_s = []
        x_s = []
        a_s = []
        real_vx_s = []
        real_va_s = []
        frames.append(0)
        vx_s.append(list[0]['vx'])
        x_s.append(list[0]['x'])
        a_s.append(3)
        real_vx_s.append(3)
        real_va_s.append(3)
        for i in range(1, len(list)):
            # print(list[i])
            object = list[i]

            vx = object['vx']
            x = object['x']

            frames.append(i)
            vx_s.append(vx)
            x_s.append(x)
            a_s.append(vx / (float(time_list[i]) - float(time_list[i - 1])))
            real_vx_s.append((x_s[i] - x_s[i - 1]) / (float(time_list[i]) - float(time_list[i - 1])))
            real_va_s.append(real_vx_s[i] / (float(time_list[i]) - float(time_list[i - 1])))
            writer.writeheader()
            writer.writerow({'No': i, 'vx': vx, 'x': x})

    # X轴，Y轴数据
    x = frames[0:30]
    y = x_s[0:30]
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, y, "r--", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("Frame(s)")  # X轴标签
    plt.ylabel("distance")  # Y轴标签
    plt.title("Line plot")  # 图标题
    plt.show()  # 显示图

    # X轴，Y轴数据
    x = frames[0:30]
    y = vx_s[0:30]
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, y, "g--", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("Frame(s)")  # X轴标签
    plt.ylabel("speed")  # Y轴标签
    plt.title("Line plot")  # 图标题
    plt.show()  # 显示图

    # X轴，Y轴数据
    x = frames[0:30]
    y = a_s[0:30]
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, y, "b--", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("Frame(s)")  # X轴标签
    plt.ylabel("accelerate")  # Y轴标签
    plt.title("Line plot")  # 图标题
    plt.show()  # 显示图

    # X轴，Y轴数据
    x = frames[5:50]
    y = real_vx_s[5:50]
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, y, "b--", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("Frame(s)")  # X轴标签
    plt.ylabel("real val speed")  # Y轴标签
    plt.title("Line plot")  # 图标题
    plt.show()  # 显示图

    # X轴，Y轴数据
    x = frames[5:50]
    y = real_va_s[5:50]
    print(y)
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, y, "b--", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("Frame(s)")  # X轴标签
    plt.ylabel("real val acceletate")  # Y轴标签
    plt.title("Line plot")  # 图标题
    plt.show()  # 显示图

    print('validate 距离: max ', max(x_s), " min ", min(x_s))
    print('validate 速度: max ', max(vx_s), " min ", min(vx_s))
    print('validate 加速度: max ', max(a_s), " min ", min(a_s))


def smoothData(video_json, time_list, x_smooth_scale=19, v_smooth_scale=20):
    pre_list = video_json['frame_data']

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

    mul = 1.15

    smooth_x_s = [x / mul for x in smooth_x_s]

    return smooth_x_s, smooth_v_s


def pltFigure(x, y, line_label, color='r--', x_label='x', y_label='x', title='fig', x_scale=1.0, x_start=1, x_end=100, y_scale=0.0, y_start=1, y_end=100):

    # plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, y, color, linewidth=1, label=line_label)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    if x_scale != 0.0:
        my_x_ticks = np.arange(x_start, x_end, x_scale)
        plt.xticks(my_x_ticks)
    if y_scale != 0.0:
        my_y_ticks = np.arange(y_start, y_end, y_scale)
        plt.yticks(my_y_ticks)
    plt.xlabel(x_label)  # X轴标签
    plt.ylabel(y_label)  # Y轴标签
    plt.title(title)  # 图标题
    plt.grid()
    plt.legend()

def smoothDistance(list):
    #参数
    max_realtive_v = 20
    max_absolute_a = 8
    t = 0.05
    beta = 0.7
    max_threshold_x = max_realtive_v * t + 2 * max_absolute_a * t * t
    max_threshold_x = 0.45
    min_threshold_x = -max_threshold_x

    smooth_x_list = [x for x in list]
    smooth_x_list.append(list[0])
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

def testVideo(start, end):
    '/Users/wangshuainan/Desktop/result/test2/'
    pre_filepath = '/Users/wangshuainan/Desktop/valid_video_01_pre.json'
    real_filepath = '/Users/wangshuainan/Desktop/mcdc_data/valid/valid_video_00_gt.json'
    with open(pre_filepath) as f:
        pre_video = json.load(f)
    with open(real_filepath) as f:
        real_video = json.load(f)

    pre_list = pre_video['frame_data']
    real_list = real_video['frame_data']

    # 将时间差读进list
    time_list = getFrameGap('/Users/wangshuainan/Desktop/mcdc_data/valid/valid_video_00_time.txt')

    frames = [i for i in range(start,end)]
    v_s = []
    x_s = []
    a_s = []
    real_x_s = []
    real_v_s = []
    real_a_s = []
    v_s.append(pre_list[0]['vx'])
    x_s.append(pre_list[0]['x'])
    a_s.append(3)
    for i in range(1, len(pre_list)):
        pre_object = pre_list[i]
        real_object = real_list[i]

        vx = pre_object['vx']
        x = pre_object['x']

        real_vx = real_object['vx']
        real_x = real_object['x']

        v_s.append(vx)
        x_s.append(x)
        a_s.append(vx / (float(time_list[i]) - float(time_list[i - 1])))

        real_x_s.append(real_x)
        real_v_s.append(real_vx)
        real_a_s.append(real_vx / (float(time_list[i]) - float(time_list[i - 1])))


    smooth_x_s = smoothDistance(x_s)

    # 绘制距离图
    plt.figure(figsize=(8, 4))
    # X轴，Y轴数据
    x = frames[start: end]
    y = x_s[start: end]
    pltFigure(x, y, 'predict_distance', 'r--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.1, min(y), max(y))
    y = real_x_s[start: end]
    pltFigure(x, y, 'real_distance', 'g--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.1, min(y), max(y))
    y = smooth_x_s[start: end]
    pltFigure(x, y, 'smooth_distance', 'b--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.1, min(y), max(y))
    plt.show()


    # smooth_v_s = []
    # smooth_scale = 25  # 取奇数
    # for i in range(0, len(pre_list)):
    #     half_window = int(smooth_scale / 2)
    #     if i < half_window:
    #         x = x_s[i + smooth_scale - 1] - x_s[i]
    #         v = x / (float(time_list[i + smooth_scale - 1]) - float(time_list[i]))
    #     elif i > len(pre_list) - half_window - 1:
    #         x = x_s[i] - x_s[i - smooth_scale + 1]
    #         v = x / (float(time_list[i]) - float(time_list[i - smooth_scale + 1]))
    #     else:
    #         x = x_s[i + half_window] - x_s[i - half_window]
    #         v = x / (float(time_list[i + half_window]) - float(time_list[i - half_window]))
    #     smooth_v_s.append(v)
    # smooth_x_s = []
    # smooth_scale = 19  # 取奇数
    # for i in range(0, len(pre_list)):
    #     half_window = int(smooth_scale / 2)
    #     if i < half_window:
    #         x = x_s[i:i + smooth_scale]
    #     elif i > len(pre_list) - half_window - 1:
    #         x = x_s[i:]
    #     else:
    #         x = x_s[i - half_window:i + half_window + 1]
    #     smooth_x = sum(x) / len(x)
    #     smooth_x_s.append(smooth_x)
    #
    # list = []
    # for i in range(20):
    #     list.append(0)
    #
    # for i in range(0, len(pre_list)):
    #     mul = smooth_x_s[i] / float(real_x_s[i])
    #     list[int(mul * 2)] += 1
    #
    # print(list)
    # print(list.index(max(list)))
    # mul = list.index(max(list)) / 2 + 0.15
    #
    # smooth_x_s = [x / mul for x in smooth_x_s]



    print('validate 距离: max ', max(x_s), " min ", min(x_s))
    print('validate 速度: max ', max(v_s), " min ", min(v_s))
    print('validate 加速度: max ', max(a_s), " min ", min(a_s))


if __name__ == "__main__":
    # /Users/wangshuainan/Desktop/mcdc_data/valid/valid_video_00_gt.json
    # /Users/wangshuainan/Desktop/valid_video_00_pre.json
    # filepath = '/Users/wangshuainan/Desktop/valid_video_00_pre.json'
    # filepath = '/Users/wangshuainan/Desktop/mcdc_data/valid/valid_video_00_gt.json'
    testVideo(0, 30)

