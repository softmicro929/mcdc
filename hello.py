import time
import random
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import darknet as dn

# prepare YOLO
dn.set_gpu(0)
net = dn.load_net(str.encode("../cfg/yolov3.cfg"),
                  str.encode("../yolov3.weights"), 0)
meta = dn.load_meta(str.encode("../cfg/coco.data"))

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

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=20)
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
        print(label, (left, top), (right, bottom))

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


def pipeline(img):
    # image data transform
    # img - cv image
    # im - yolo image
    im = array_to_image(img)
    dn.rgbgr_image(im)

    tic = time.time()
    result = dn.detect1(net, meta, im)

    #print(result)

    toc = time.time()
    print(toc - tic, result)

    img_final = draw_boxes(img, result)
    return img_final, result


count_frame, process_every_n_frame = 0, 1
# get camera device
#cap = cv2.VideoCapture(0)

video = cv2.VideoCapture('/Users/wangshuainan/Desktop/mcdc_data/valid/valid_video_00.avi')
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video.get(cv2.CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

#video.read()
while(True):
    # get a frame
    
    ret, frame = video.read()
    count_frame += 1
        
    # show a frame
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize image half
    cv2.imshow("Video", img)

    #if running slow on your computer, try process_every_n_frame = 10
    if count_frame % process_every_n_frame == 0:
        cv2.imshow("YOLO", pipeline(img))

    # press keyboard 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
cv2.destroyAllWindows()
