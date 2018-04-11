# -*- encoding:utf-8 -*-
import os
import sys
import os.path
import CONFIG as CONFIG
def list_file(path):
    count = 0
    video_list=[]
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == '.avi': 
            video_list.append(filename)
            print(filename) 
    # 将路径下的
    for video in video_list:
        handleVideo(video)
    #写json有python完成



if __name__=="__main__":
    list_file(CONFIG.TEST_VIDEO_PATH)