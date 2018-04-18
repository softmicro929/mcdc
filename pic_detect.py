# -*- encoding:utf-8 -*-
import time
import random
import colorsys
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import darknet as dn
import bird_view_projection as birdView
import json
import CONFIG_SERVER_TEST as CONFIG
import smooth as smooth

# prepare YOLO
dn.set_gpu(0)
net = dn.load_net(str.encode(CONFIG.DARKNET_DIR + "cfg/yolov3.cfg"),
                  str.encode(CONFIG.DARKNET_DIR + "yolov3.weights"), 0)
meta = dn.load_meta(str.encode(CONFIG.DARKNET_DIR + "cfg/coco.data"))

# box colors
box_colors = None


def generate_colors(num_classes):
    global box_colors

    if box_colors != None and len(box_colors) > num_classes:
        return box_colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    box_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    box_colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            box_colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    # Shuffle colors to decorrelate adjacent classes.
    random.shuffle(box_colors)
    random.seed(None)  # Reset seed to default.


def draw_boxes(img, result):
    image = Image.fromarray(img)

    font = ImageFont.truetype(str.encode(CONFIG.DARKNET_DIR + 'font/FiraMono-Medium.otf'), 20)
    thickness = (image.size[0] + image.size[1]) // 300

    num_classes = len(result)
    generate_colors(num_classes)

    index = 0
    for objection in result:
        index += 1
        class_name, class_score, (x, y, w, h) = objection
        # print(name, score, x, y, w, h)

        left = int(x - w / 2)
        right = int(x + w / 2)
        top = int(y - h / 2)
        bottom = int(y + h / 2)

        label = '{} {:.2f}'.format(class_name.decode('utf-8'), class_score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        # print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i,
                            bottom - i], outline=box_colors[index - 1])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=box_colors[index - 1])
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        del draw

    return np.array(image)


def array_to_image(arr):
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr / 255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w, h, c, data)
    return im


def drawBoxOnImg(img, x, y, w, h, p_x, p_y, num):
    # img图像，起点坐标，终点坐标（在这里是x+w,y+h,因为w,h分别是人脸的长宽）颜色，线宽）
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (127, 255, 0), 5)
    cv2.circle(img, (int(p_x), int(p_y)), 5, (255, 0, 0), -1)
    img_path = CONFIG.DARKNET_DIR + 'pic/' + str(num) + '.png'
    cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])


def pipeline(img):
    # image data transform
    # img - cv image
    # im - yolo image
    im = array_to_image(img)
    dn.rgbgr_image(im)

    tic = time.time()
    result = dn.detect1(net, meta, im)

    toc = time.time()
    # print('------------------------pipeline:',toc - tic, result)
    img_final = draw_boxes(img, result)
    # img_final = img
    return img_final, result


# 设计算法：得到车的json数组， 找出最可能是前车的框
# // 1、remove json不是车部分
# // 2、除去距离底边最近的车前身 (可以联合相机参数camera_front front越大，剔除的范围越大),
# // 2.5 如果自己的训练的时候，剔除那些被遮挡的车，被遮挡部分大于20% 不需要这个参数了，交并
# // 3、读取相机参数， left,right, 找出相机处于图片的位置y_camera    left+right = 车身长度   left和right大概是从车前方看的 从摄像头角度看要反过来
# // 4、将那些车按照y距离y_camera的距离d=ABS(（边框左下角y+边框右下角y）/2-y_camera)排序，
# // 5、选择d最小的框
# // 缺点，没有考虑道路方向，也许会变化先这么设定吧.
# [
#   (b'bicycle', 0.9941766262054443,
#       (363.4241638183594, 278.7040100097656, 396.94329833984375, 331.8062438964844)),
#   (b'dog', 0.9900424480438232,
#       (221.59780883789062, 380.45477294921875, 186.77037048339844, 312.46099853515625)),
#   (b'truck', 0.9237195253372192,
#       (581.048583984375, 128.2719268798828, 215.67906188964844, 85.07489776611328))
# ]

def chooseOne(list, cam):
    if list is None:
        return None  # 再说
    i = 0
    while i < len(list):
        iterater = list[i]
        # print(iterater)
        if iterater[0] != b'car' and iterater[0] != b'truck' and iterater[0] != b'bus':
            list.remove(iterater)
        i = i + 1

    if len(list) == 0:
        return None  # 再说

    width = float(cam['image_width'])
    left = float(cam['cam_to_right'])
    right = float(cam['cam_to_left'])
    y_camera = width * left / (left + right)  # rough
    list = sorted(list, key=lambda x: abs(y_camera - (x[2][0] + x[2][2] / 2)))
    return list[0]

# 1
def chooseOnImprove(pic_list):
    if pic_list is None:
        return None  # 再说

    width = float(2304)
    height = float(1296)
    left = 0.7
    right = 1.0
    x_car_mid = (width * left / (left + right) + width / 2) / 2  # 加上中点平滑处理一下 有待改进
    # x_car_mid=width/2
    # x_car_mid= width*left/(left+right)/5 +width*2/5

    i = 0
    while i < len(pic_list):
        iterater = pic_list[i]
        # print(iterater)
        # 中心点，宽度，高度
        p0 = iterater[2][0]
        p1 = iterater[2][1]
        w = iterater[2][2]
        h = iterater[2][3]

        if not (iterater[0] == b'car' or iterater[0] == b'truck' or iterater[0] == b'bus'):
            pic_list.remove(iterater)
            continue
        # elif h / w > 1.4:
        #     pic_list.remove(iterater)
        #     continue
        elif abs(x_car_mid - p0) > width / 5:
            pic_list.remove(iterater)
            continue
        # elif p1 > height * 0.9 and w > width*0.8:
        elif p1 + h / 2 > height * 0.92 and w > width * 0.7 and h < height * 0.4:
            pic_list.remove(iterater)
            continue
        i = i + 1

    if len(pic_list) == 0:
        return None  # 再说

    pic_list = sorted(pic_list, key=lambda x: -x[2][1])
    print('----------------------choose', pic_list[0])
    return pic_list[0]



def chooseOneImproveWithTracking(pic_list):
    if pic_list is None:
        return None  # 再说

    width = float(2304)
    height = float(1296)
    left = 0.7
    right = 1.0
    x_car_mid = (width * left / (left + right) + width / 2) / 2  # 加上中点平滑处理一下 有待改进
    # x_car_mid=width/2
    # x_car_mid= width*left/(left+right)/5 +width*2/5

    i = 0
    while i < len(pic_list):
        iterater = pic_list[i]
        # print(iterater)
        # 中心点，宽度，高度
        p0 = iterater[2][0]
        p1 = iterater[2][1]
        w = iterater[2][2]
        h = iterater[2][3]

        if not (iterater[0] == b'car' or iterater[0] == b'truck' or iterater[0] == b'bus'):
            pic_list.remove(iterater)
            continue
        # elif h / w > 1.4:
        #     pic_list.remove(iterater)
        #     continue
        elif abs(x_car_mid - p0) > width / 5:
            pic_list.remove(iterater)
            continue
        # elif p1 > height * 0.9 and w > width*0.8:
        elif p1 + h / 2 > height * 0.92 and w > width * 0.7 and h < height * 0.4:
            pic_list.remove(iterater)
            continue
        i = i + 1

    if len(pic_list) == 0:
        return None  # 再说

    pic_list = sorted(pic_list, key=lambda x: -x[2][1])
    print('----------------------choose', pic_list[0])
    return pic_list[0]

def chooseBBoxImprove_line_colapse(pic_list):
    if pic_list is None:
        return None  # 再说

    width = float(2304)
    height = float(1296)
    left = 0.7
    right = 1.0
    x_car_mid = (width * left / (left + right) + width / 2) / 2  # 加上中点平滑处理一下 有待改进
    # x_car_mid=width/2
    # x_car_mid= width*left/(left+right)/5 +width*2/5

    i = 0
    while i < len(pic_list):
        iterater = pic_list[i]
        # print(iterater)
        # 中心点，宽度，高度
        p0 = iterater[2][0]
        p1 = iterater[2][1]
        w = iterater[2][2]
        h = iterater[2][3]

        if not (iterater[0] == b'car' or iterater[0] == b'truck' or iterater[0] == b'bus'):
            pic_list.remove(iterater)
            continue
        # elif h / w > 1.4:
        #     pic_list.remove(iterater)
        #     continue
        elif abs(x_car_mid - p0) > width / 5:
            pic_list.remove(iterater)
            continue
        # elif p1 > height * 0.9 and w > width*0.8:
        elif p1 + h / 2 > height * 0.92 and w > width * 0.7 and h < height * 0.4:
            pic_list.remove(iterater)
            continue
        i = i + 1

    if len(pic_list) == 0:
        return None  # 再说

    #pic_list = sorted(pic_list, key=lambda x: -x[2][1])
    pic_list = sorted(pic_list, key=lambda x: abs(x[2][1]-x_car_mid))
    # sort by distants to car_mid_line

    res=-1
    l=len(pic_list)
    for i in range(l):
        flag= True
        bboxi=pic_list[i]
        for j in range(i+1,l):
            bboxj=pic_list[j]
            # if no collapse ,stil true; else flag=false
            if judgeOk(bboxi, bboxj):
                continue
            else:
                flag = False
                break
        if flag:
            res = i
            break

    print('----------------------choose', pic_list[0])
    if res != -1:
        return pic_list[res]
    else:
        pic_list = sorted(pic_list, key=lambda x: -x[2][1])
        print('----------------------choose', pic_list[0])
        return pic_list[0]

def judgeOk(bboxi, bboxj):
    if bboxi[2][1]+bboxi[2][3]/2 > bboxj[2][1]+bboxj[2][3]/2:
        return True

    if bboxj[2][0]>bboxi[2][0]+bboxi[2][2] or bboxj[2][0]+bboxj[2][2]<bboxi[2][0]:
        return True

    return False



def chooseOneWithWeight(pic_list):
    #   1/distant to midLine  2/distant to loweres line
    #   3/angle to mid point    4/area of rectangle     5/IOU between rectangles
    if pic_list is None:
        return None  # 再说

    width = float(2304)
    height = float(1296)
    left = 0.7
    right = 1.0
    #x_car_mid = (width * left / (left + right) + width / 2) / 2  # 加上中点平滑处理一下 有待改进
    x_car_mid = width*left/(left+right)/5 + width*2/5

    i = 0
    while i < len(pic_list):
        iterater = pic_list[i]
        # print(iterater)
        # 中心点，宽度，高度
        p0 = iterater[2][0]
        p1 = iterater[2][1]
        w = iterater[2][2]
        h = iterater[2][3]

        # if iterater[0]!='car' and iterater[0]!='truck' and iterater[0]!='bus':
        if not (iterater[0] == b'car' or iterater[0] == b'truck' or iterater[0] == b'bus'):
            pic_list.remove(iterater)
            continue
        elif abs(x_car_mid - p0) > width / 4:  # too far from mid line
            pic_list.remove(iterater)
            continue
        # elif p1 > height * 0.9 and w > width*0.8:
        elif p1+h/2 > height * 0.9 and w > width*0.6 and h < 0.4 * height:
            pic_list.remove(iterater)
            continue
            # 和车中点距离过于远：
        i = i + 1

    #
    if len(pic_list) == 0:
        return None  # 再说

    # y越大越可能是前车
    pic_list = sorted(pic_list, key=lambda x: -x[2][1])
    print('----------------------choose', pic_list[0])
    return pic_list[0]


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



def handlePic(video_path, time_txt_name, output_result_json_path, camera_param_json_name):
    video = cv2.VideoCapture(video_path)
    print('------------open video')
    # # Find OpenCV version
    # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    #
    # if int(major_ver)  < 3 :
    #     fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    #     print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    # else :
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    birdView.setCameraParams(camera_param_json_name)

    with open(camera_param_json_name, 'r') as f:
        temp = json.loads(f.read())

    count_frame, process_every_n_frame = 0, 1

    # "前一帧在图像中选取的点"，用来避免当前帧没有框出车
    pre_x = 960.00
    pre_y = 960.00
    pre_dis_x = 0

    pre_box_x = 0
    pre_box_y = 0
    pre_box_w = 0
    pre_box_h = 0

    # 将时间差读进list
    time_list = getFrameGap(time_txt_name)

    # print(time_list)
    print('------------read time_txt finished, lines:', len(time_list), time_txt_name)

    result_list = []

    car_list = ['/Users/wangshuainan/Desktop/image/1523465188473.jpg',
                '/Users/wangshuainan/Desktop/image/1523465217730.jpg',
                '/Users/wangshuainan/Desktop/image/1523465247087.jpg']
    i = 0

    while (True):
        # if count_frame % 10 == 0:
        print('-----------------------count_frame:', count_frame)
        # if count_frame > 300:
        #     break
        # get a frame
        ret, img = video.read()
        # if i < 3:
        #     img = cv2.imread(car_list[i])
        #     i += 1
        # else:
        #     break
        # if img is None:
        #     print("video.read() fail || video.read() is end!")
        #     break
        if img is None or ret is None:
            print("video.read() fail || video.read() is end!")
            break

        # show a frame
        # img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize image half
        # cv2.imshow("Video", img)

        # if running slow on your computer, try process_every_n_frame = 10
        if count_frame % process_every_n_frame == 0:
            # cv2.imshow("YOLO", pipeline(img))
            #  res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
            image_, boxes = pipeline(img)
            only_box = chooseOnImprove(boxes, temp)

            # 如果定位框返回空的话，用前面的框
            if only_box is not None:
                # find target box
                # return a tuple : (b'truck', 0.9237195253372192, (581.048583984375, 128.2719268798828, 215.67906188964844, 85.07489776611328))
                class_name, class_score, (x, y, w, h) = only_box
                # print(name, score, x, y, w, h)
                left = int(x - w / 2)
                right = int(x + w / 2)
                top = int(y - h / 2)
                bottom = int(y + h / 2)

                x1, y1 = x, y + int(h / 2)

                box_x = int(x - w / 2)
                box_y = int(y - h / 2)
                box_w = w
                box_h = h

                pre_x, pre_y = x1, y1
                pre_box_x, pre_box_y, pre_box_w, pre_box_h = box_x, box_y, box_w, box_h
                print('------------------only_box is Not null:', x1, y1)

            else:
                x1, y1 = pre_x, pre_y
                box_x, box_y, box_w, box_h = pre_box_x, pre_box_y, pre_box_w, pre_box_h
                print('------------------only_box is null', x1, y1)

            # drawBoxOnImg(img, box_x, box_y, box_w, box_h, x1, y1, i)

            # 然后计算速度+距离
            # distance_x代表相距前车距离
            print('--------------------------birdView.getXY---')
            distance_x, distance_y = birdView.getXY(x1, y1)
            print('--------------------------birdView.getXY---', distance_x, distance_y)
            if count_frame > 0:
                speed_x = (distance_x - pre_dis_x) / float(
                    float(time_list[count_frame]) - float(time_list[count_frame - 1]))
            else:
                # 第一帧的速度默认为10m/s,然后最后输出时再用第二帧的速度去校正它
                speed_x = 10
            pre_dis_x = distance_x
            pre_speed_x = speed_x

            # test_video_00_pre.json
            #  {
            #  "vx": -2.3125, //相对速度
            #  "x": 11.0625, //相对位置
            #  "fid": 0 //frame_id, 帧号，输出时帧号从 0 开始顺序依次递增
            #  }
            # }
            dict = {'vx': speed_x, 'x': distance_x, "fid": i,
                    'ref_bbox': {"top": box_y, "right": box_x + box_w,
                                 "bot": box_y + box_h, "left": box_x}}

            result_list.append(dict)
            count_frame += 1
            i += 1

    smooth_result_list = smooth.smoothData(result_list, time_list)
    print('=========pipeline finished,result============>')
    print(smooth_result_list)
    print('=============================================>')

    # DO YOUR JSON CONV JOB!!!
    final_dict = {'frame_data': smooth_result_list}

    # # with open不用考虑关闭流和异常
    # with open(output_result_json_path, 'w') as json_file:
    #     json.dump(final_dict, json_file, ensure_ascii=False)

    # video_path传入 封装成函数后改成下面的
    with open(output_result_json_path, 'w+') as json_file:
        json.dump(final_dict, json_file, ensure_ascii=False)

    print('=========pipeline finished,write json finished============>')
    video.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":

    car_list = []
    for filename in os.listdir('/home/m10/workspace/darknet/originpic/cpytest/'):
        car_list.append('/home/m10/workspace/darknet/originpic/cpytest/'+filename)
    print(len(car_list))
    i = 0
    while i < len(car_list):
        img = cv2.imread(car_list[i])

        image_, boxes = pipeline(img)
        cv2.imwrite('../pic/' + str(i) + '_all.png', image_, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        # only_box = chooseOnImprove(boxes, temp)
        only_box = chooseOnImprove(boxes)
        if only_box is None:
            print('list'+str(i) + ':' + car_list[i]+' has no box')
            i += 1
            continue
        class_name, class_score, (x, y, w, h) = only_box
        # print(name, score, x, y, w, h)
        left = int(x - w / 2)
        right = int(x + w / 2)
        top = int(y - h / 2)
        bottom = int(y + h / 2)

        x1, y1 = x, y + int(h / 2)

        box_x = int(x - w / 2)
        box_y = int(y - h / 2)
        box_w = w
        box_h = h

        drawBoxOnImg(img, box_x, box_y, box_w, box_h, x1, y1, i)

        i += 1

    # lenna_img = cv2.imread("../data/8218.png")
    # img,_ = pipeline(lenna_img)
    # cv2.imwrite('../pic/2.jpg',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    # cv2.imshow("YOLO", img)

    # lenna_img = cv2.imread("../data/8219.png")
    # img,_ = pipeline(lenna_img)
    # cv2.imwrite('../pic/3.jpg',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    # cv2.imshow("YOLO", img)
    # cv2.waitKey(1)

