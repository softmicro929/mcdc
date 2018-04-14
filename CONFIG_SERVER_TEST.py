# -*- encoding:utf-8 -*-
# VALID_VIDEO_PATH = '/data/mcdc_data/02/valid/'
#
# CAMERA_PARAMETER_PATH = '/data/mcdc_data/02/valid/camera_parameter.json'
#
# WRITE_JSON_DIR='/home/m10/workspace/darknet/mcdc/'
#
# TEST_VIDEO_DIR='/data/mcdc_data/02/valid/'




# VALID_VIDEO_PATH = '/data/mcdc_data/valid/'
#
# CAMERA_PARAMETER_PATH = '/data/mcdc_data/valid/camera_parameter.json'
#
# WRITE_JSON_DIR='/home/m10/workspace/darknet/mcdc/'
#
# TEST_VIDEO_DIR='/data/mcdc_data/valid/'
#
# DARKNET_DIR= '/home/m10/workspace/darknet/'

VALID_VIDEO_PATH = '/data/mcdc_data/test/'
CAMERA_PARAMETER_PATH = '/data/mcdc_data/test/camera_parameter.json'
WRITE_JSON_DIR='/home/m10/test_pre/'
TEST_VIDEO_DIR='/data/mcdc_data/test/'
DARKNET_DIR= '/home/m10/workspace/darknet/'


# myself test
# VALID_VIDEO_PATH = '/home/m10/mytest/valid/'
# CAMERA_PARAMETER_PATH = '/data/mcdc_data/valid/camera_parameter.json'
# WRITE_JSON_DIR='/home/m10/mytest/valid/'
# TEST_VIDEO_DIR='/home/m10/mytest/valid/'
# DARKNET_DIR= '/home/m10/workspace/darknet/'


#测试数据所在路径、|||||||||||||||||||||||||||||||||

#要写的json文件的路径

# 服务器上自己测试的


# 工程项目文件全部统一放在 ~/MCDC_队伍名/ 中，并写好 run.sh 一键运行
# run.sh 放在~/MCDC_Soft_Micro/
# 算法接口。一键运行程序会读取/data/mcdc_data/test/ 中的所有 video 及其时间戳
# 文件并将结果写到~/test_pre/ 中。test/ 的结构组织类比 valid/，除了 gt.json 没有
# 外其他都一致。~/test_pre/ 中放上队伍名.txt，写上自己对应的英文队伍名；所
# 有参赛队伍对测试视频不可访问。具体如下：
# 训练、评估文件夹组织形式 测试文件夹组织形式
# /data/mcdc_data/valid/
# ---- valid_video_00.avi
# ---- valid_video_01.avi
# ---- valid_video_00_gt.json
# ---- valid_video_01_gt.json
# ---- valid_video_00_time.txt
# ---- valid_video_01_time.txt
# ---- camera_parameter.json
# ~/MCDC_队伍名/
# ---- run.sh
# ---- 选手代码 project
# ~/test_pre/
# ---- 队伍名.txt
# ---- test_video_00_pre.json
# ---- test_video_01_pre.json
# /data/mcdc_data/test/
# ---- test_video_00.avi
# ---- test_video_01.avi
# ---- test_video_00_time.txt
# ---- test_video_01_time.txt
# ---- camera_parameter.j