# -*- encoding:utf-8 -*-
import os
import sys
import os.path
import CONFIG_SERVER_TEST as CONFIG
import hello as h
def list_file():
    count = 0
    video_list=[]
    for filename in os.listdir(CONFIG.TEST_VIDEO_DIR):
        if os.path.splitext(filename)[1] == '.avi': 
            video_list.append(filename)
            print(filename) 

    for videoname in video_list:

        time_txt_name = CONFIG.TEST_VIDEO_DIR+videoname[0:-4]+'_time.txt' #时间文件名
        json_name     = CONFIG.WRITE_JSON_DIR+videoname[0:-4]+'_pre.json'
        video_path    = CONFIG.TEST_VIDEO_DIR+videoname
        h.handleVideo(video_path,time_txt_name,json_name, CONFIG.CAMERA_PARAMETER_PATH)


if __name__=="__main__":
    list_file()