#-------------------------------------#
#       调用yolo_save_mAP.py下的YOLO类，
#       注意要更改yolo_save_mAP里面的类别和入模型的尺寸
#-------------------------------------#
from yolo_save_mAP import YOLO
from PIL import Image
import numpy as np
import cv2, os
import time

#use which gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

yolo = YOLO()

def detect_save_image(image_path, savetxt_path):
    '''
    检测一个文件夹内所有图片并保存至指定路径
    :param image_path: 测试图片路径
    :param savetxt_path: 保存txt结果路径
    :return:
    '''
    for img_name in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        # 格式转变，BGRtoRGB
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_RGB_iter = Image.fromarray(np.uint8(image_RGB))
        # 把结果写入txt
        f = open(savetxt_path + '/' + img_name.split('.')[0] + '.txt', "w", encoding='utf-8')
        # 进行检测
        print('current image : ', img_name)
        line = yolo.detect_image(image_RGB_iter)
        # detect_img = np.array(detect_img)
        f.write(line)
        f.close()
        # # RGBtoBGR满足opencv显示格式
        # detect_img_BGR = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(save_path + '/' + img_name, detect_img_BGR)

if __name__ == '__main__':
    img_path = r'/usr/idip/idip/liuan/data/make_Object_Detect_format_dataset/4class_test/jpg'
    save_path = r'/usr/idip/idip/liuan/data/save_txt/4class/AP50/yolov4_608*608_epoch150'

    # img_path = r'/usr/idip/idip/liuan/project/yolov4-pytorch/test_file/img_path'
    # save_path = r'/usr/idip/idip/liuan/project/yolov4-pytorch/test_file/save_txt_path'
    detect_save_image(img_path, save_path)
