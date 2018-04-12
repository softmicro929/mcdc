# -*- encoding:utf-8 -*-
import os
import sys
import os.path
import CONFIG_SERVER_TEST as CONFIG
def list_file(path):
    count = 0
    video_list=[]
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == '.avi': 
            video_list.append(filename)
            print(filename) 

    for videoname in video_list:

        time_txt_name = CONFIG.TEST_VIDEO_DIR+videoname[0:-4]+'_time.txt' #时间文件名
        json_name     = CONFIG.WRITE_JSON_DIR+videoname[0:-4]+'_pre.json'
        video_path    = CONFIG.TEST_VIDEO_DIR+videoname
        handleVideo(video_path,time_txt_name,json_name, CAMERA_PARAMETER_PATH)
        #写json由python完成

# 工程项目文件全部统一放在 ~/MCDC_队伍名/ 中，并写好 run.sh 一键运行
# 算法接口。一键运行程序会读取/data/mcdc_data/test/ 中的所有 video 及其时间戳
# 文件并将结果写到~/test_pre/ 中。test/ 的结构组织类比 valid/，除了 gt.json 没有
# 外其他都一致。~/test_pre/ 中放上队伍名.txt，写上自己对应的英文队伍名；所
# 有参赛队伍对测试视频不可访问。具体如下：


if __name__=="__main__":
    list_file(CONFIG.TEST_VIDEO_DIR)