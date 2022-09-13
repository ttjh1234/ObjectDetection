import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from time import *
from .neptune import record_image
from .data import inverse_trans


def vision_valid(image,gt_box,run,visable=0,file_save=0):
    img_rgb_copy = image.numpy().copy()/255.0
    green_rgb = (125, 255, 51)
    for rect in gt_box:

        left = rect[1]*image.shape[1]
        top = rect[0]*image.shape[0]
        # rect[2], rect[3]은 너비와 높이이므로 우하단 좌표를 구하기 위해 좌상단 좌표에 각각을 더함.
        right = rect[3]*image.shape[1]
        bottom = rect[2]*image.shape[0]
        

        img_rgb_copy = cv2.rectangle(img_rgb_copy, (int(left), int(top)), (int(right), int(bottom)), color=green_rgb, thickness=1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb_copy)
    plt.show()
    if visable==1:
        img=(img_rgb_copy*255).astype("uint8")
        img=Image.fromarray(img)
        record_image(run,img)
    
    if file_save==1:
        img=(img_rgb_copy*255).astype("uint8")
        img=Image.fromarray(img)
        name=str(int(time()))+'.png'
        while name in os.listdir("C:/Users/UOS/Desktop/result/"):
            name=str(int(time()))+'.png'
        img.save("C:/Users/UOS/Desktop/result/"+name)
        

# RPN Network Performance test
def valid_result2(valid,rpn_model,anchor_box,iou=0.3,max_n=300,visable=0,file_save=0):
    valid_reg,valid_cls=rpn_model(tf.expand_dims(valid["image"],axis=0),training=False)
    print("GT Results")
    vision_valid(valid["image"],valid["bbox"],visable)
    print("Model Results")
    valid_reg2=inverse_trans(anchor_box,valid_reg)
    valid_reg3=tf.reshape(valid_reg2,(-1,4))
    # anchor_box.shape vs valid_reg.shape
    v_cls=tf.reshape(valid_cls,(-1))
    t2=tf.where(v_cls>=0.5,v_cls,0.0)
    proposed_box=tf.image.non_max_suppression_with_scores(valid_reg3,t2,max_n,iou_threshold=iou)
    fgind=tf.where(proposed_box[1]>=0.5)
    v_pdata = tf.gather(valid_reg3, proposed_box[0])
    v_pdata= tf.gather_nd(v_pdata,indices=fgind)
    vision_valid(valid["image"],v_pdata,visable,file_save)