# -*- encoding:utf-8 -*-
VALID_VIDEO_PATH = '/data/mcdc_data/valid/'
CAMERA_PARAMETER_PATH = '/Users/wangshuainan/Desktop/mcdc_data/valid/camera_parameter.json'

FRAME_GAP_TIME_PATH = '/Users/wangshuainan/Desktop/mcdc_data/valid/valid_video_00_time.txt'

WRITE_JSON_PATH=''

TEST_VIDEO_PATH='/data/mcdc_data/test/' 
#测试数据所在路径
TEST_DIR_official= '~/test_pre/' 
TEST_DIR_test= '~/test_pre/' 
#要写的json文件的路径




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