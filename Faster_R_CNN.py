# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:19:01 2022

@author: Sungsu
"""

# Faster RCNN Before Modularization

# Library list

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
from tqdm import tqdm

# Patch Data

dataset,info=tfds.load("voc",with_info=True)
info.features['labels'].names

voc_train,voc_test,voc_valid=dataset["train"],dataset["test"],dataset["validation"]

# Dat preprocess  
# resize the image (500,500,3) and extract gt_box

def data_preprocess(feature):
  img=feature['image']
  #label=feature['labels']
  bbox=feature['objects']['bbox']
  #paddings = [[0, 100-tf.shape(bbox)[0]], [0, 0]]
  #bbox = tf.pad(bbox, paddings, 'CONSTANT', constant_values=0)
  #bbox=tf.reshape(bbox,(100,4))
  image=tf.image.resize(img,[500,500])
  return {"image":image,"bbox":bbox}

voc_train2=voc_train.map(lambda feature: data_preprocess(feature))
voc_test2=voc_test.map(lambda feature: data_preprocess(feature))
voc_valid2=voc_valid.map(lambda feature: data_preprocess(feature))

# Utils

# make_anchor : generating anchor box 
# Paper propose : {scale : [128,256,512], ratio : {0.5, 1, 2}} 
# anchor shape : {31,31,9,4}

def make_anchor():
  width,height=31,31
  x=tf.range(width) # width는 특성맵의 width
  y=tf.range(height) # height는 특성맵의 height
  X,Y=tf.meshgrid(x,y)
  center_x=tf.math.add(tf.cast(X,tf.float32),0.5)/31
  center_y=tf.math.add(tf.cast(Y,tf.float32),0.5)/31
  
  scale=[128,256,512]
  ratio1=[0.5,1,2]
  anchor_wh=[]
  for i in scale:
    for r1 in ratio1:
      w=tf.sqrt((i/500)**2/r1) # width는 원래 input image의 width : 500 가정
      h=w*r1 # height는 원래 input image의 height : 500 가정
      anchor_wh.append([w/2,h/2])

  for i in range(len(anchor_wh)):
    xmin=tf.clip_by_value(center_x-anchor_wh[i][0], 0, 1)
    xmax=tf.clip_by_value(center_x+anchor_wh[i][0], 0, 1)
    ymin=tf.clip_by_value(center_y-anchor_wh[i][1], 0, 1)
    ymax=tf.clip_by_value(center_y+anchor_wh[i][1], 0, 1)
    if i==0:
      anchor_box=tf.stack([ymin,xmin,ymax,xmax],axis=2)
      anchor_box=tf.expand_dims(anchor_box,axis=3)
    else:
      temp=tf.stack([ymin,xmin,ymax,xmax],axis=2)
      temp=tf.expand_dims(temp,axis=3)
      anchor_box=tf.concat([anchor_box,temp],axis=3)
  
  anchor_box=tf.transpose(anchor_box,[0,1,3,2])
  return anchor_box


# making_rpn_train : Old preprocess


def making_rpn_train(anchor_box,gt_box):
  # anchor_box와 이미지 내 gt_box와의 IoU 계산
  # IoU가 0.7 이상이면 positive sample 0.3 미만이면 negative sample
  
  gt_box_size=(gt_box[:,2]-gt_box[:,0])*(gt_box[:,3]-gt_box[:,1])
  anchor_box=tf.reshape(anchor_box,[31*31*9,1,4])
  anchor_box_size=(anchor_box[:,:,2]-anchor_box[:,:,0])*(anchor_box[:,:,3]-anchor_box[:,:,1])
  
  xmin=tf.math.maximum(anchor_box[:,:,1],gt_box[:,1])
  ymin=tf.math.maximum(anchor_box[:,:,0],gt_box[:,0])
  xmax=tf.math.minimum(anchor_box[:,:,3],gt_box[:,3])
  ymax=tf.math.minimum(anchor_box[:,:,2],gt_box[:,2])

  intersection=(xmax-xmin)*(ymax-ymin)
  intersection2=tf.where((xmax>xmin)&(ymax>ymin),intersection,0)
  intersection3=tf.where(intersection2>0,intersection2,0)

  union=gt_box_size[tf.newaxis,:]+anchor_box_size-intersection3
  iou=intersection3/union

  # 추가로 해야 할 부분
  # 도출된 iou값을 기준으로 index를 알아와서 positive와 negative 구분하기.
  positive_index=tf.where(iou>=0.6)
  iou_positive=tf.where(iou>=0.6)
  negative_index=tf.where(iou<0.3)
  iou_negative=tf.where(iou<0.3)

  positive_ind=tf.stack([(positive_index[:,0]//9)//31,(positive_index[:,0]//9)%31,positive_index[:,0]%9,iou_positive[:,1]],axis=1)
  negative_ind=tf.stack([(negative_index[:,0]//9)//31,(negative_index[:,0]//9)%31,negative_index[:,0]%9,iou_negative[:,1]],axis=1)

  return iou,positive_ind,negative_ind


# making_valid_positive : For vision model performance, preprocess valid_set
# Not batch process

def making_valid_positive(anchor_box,gt_box):
  gt_box_size=(gt_box[:,2]-gt_box[:,0])*(gt_box[:,3]-gt_box[:,1])
  anchor_box=tf.reshape(anchor_box,[31*31*9,1,4])
  anchor_box_size=(anchor_box[:,:,2]-anchor_box[:,:,0])*(anchor_box[:,:,3]-anchor_box[:,:,1])
  print(gt_box_size.shape,anchor_box_size.shape)

  xmin=tf.math.maximum(anchor_box[:,:,1],gt_box[:,1])
  ymin=tf.math.maximum(anchor_box[:,:,0],gt_box[:,0])
  xmax=tf.math.minimum(anchor_box[:,:,3],gt_box[:,3])
  ymax=tf.math.minimum(anchor_box[:,:,2],gt_box[:,2])

  intersection=(xmax-xmin)*(ymax-ymin)
  intersection2=tf.where((xmax>xmin)&(ymax>ymin),intersection,0)
  intersection2=tf.where((xmax<xmin)|(ymax<ymin),0,intersection2)
  intersection3=tf.where(intersection2>0,intersection2,0)
  print(intersection3.shape)

  union=gt_box_size[tf.newaxis,:]+anchor_box_size-intersection3
  print(union.shape)
  iou=intersection3/union

  # 추가로 해야 할 부분
  # 도출된 iou값을 기준으로 index를 알아와서 positive와 negative 구분하기.
  positive_index=tf.where(iou>=0.6)
  iou_positive=tf.where(iou>=0.6)


  positive_ind=tf.stack([(positive_index[:,0]//9)//31,(positive_index[:,0]//9)%31,positive_index[:,0]%9,iou_positive[:,1]],axis=1)
  print(positive_ind.shape)
  return positive_ind


# vision_valid : print image and bbox

def vision_valid(image,gt_box):
  img_rgb_copy = image.numpy().copy()/255.0
  green_rgb = (125, 255, 51)
  for rect in gt_box:

    left = rect[1]*image.shape[1]
    top = rect[0]*image.shape[0]
    # rect[2], rect[3]은 너비와 높이이므로 우하단 좌표를 구하기 위해 좌상단 좌표에 각각을 더함.
    right = rect[3]*image.shape[1]
    bottom = rect[2]*image.shape[0]
    

    img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color=green_rgb, thickness=1)

  plt.figure(figsize=(8, 8))
  plt.imshow(img_rgb_copy)
  plt.show()


#  making_loss_data : For vision model performance, preprocess valid set

def making_loss_data(y_true,y_pred,anchor):
  w_a=anchor[:,3]-anchor[:,1]
  h_a=anchor[:,2]-anchor[:,0]
  t_x_star=y_true[:,1]-anchor[:,1]/w_a
  t_x=y_pred[:,1]-anchor[:,1]/w_a
  t_y_star=y_true[:,0]-anchor[:,0]/h_a
  t_y=y_pred[:,0]-anchor[:,0]/h_a
  t_w_star=tf.math.log((y_true[:,3]-y_true[:,1])/w_a)
  t_w=tf.math.log(tf.clip_by_value((y_pred[:,3]-y_pred[:,1]),1e-4,1e+10)/w_a)
  t_h_star=tf.math.log((y_true[:,2]-y_true[:,0])/h_a)
  t_h=tf.math.log(tf.clip_by_value((y_pred[:,2]-y_pred[:,0]),1e-4,1e+10)/h_a)

  return tf.stack([t_x_star,t_y_star,t_w_star,t_h_star],axis=1),tf.stack([t_x,t_y,t_w,t_h],axis=1)



def non_maximum_suppression(b_box,confidence_score):
  # Plan to implement
  return None


# patch_batch : data preprocess for rpn model
# usage : tfds map function 

def patch_batch(data):
  image=data['image']
  gt_box=data['bbox']

  anchor_box=make_anchor()
  anchor=anchor_box

  p_flag=0

  gt_box_size=(gt_box[:,2]-gt_box[:,0])*(gt_box[:,3]-gt_box[:,1])
  anchor_box=tf.reshape(anchor_box,[31*31*9,1,4])
  anchor_box_size=(anchor_box[:,:,2]-anchor_box[:,:,0])*(anchor_box[:,:,3]-anchor_box[:,:,1])
  
  xmin=tf.math.maximum(anchor_box[:,:,1],gt_box[:,1])
  ymin=tf.math.maximum(anchor_box[:,:,0],gt_box[:,0])
  xmax=tf.math.minimum(anchor_box[:,:,3],gt_box[:,3])
  ymax=tf.math.minimum(anchor_box[:,:,2],gt_box[:,2])

  intersection=(xmax-xmin)*(ymax-ymin)
  intersection2=tf.where((xmax>xmin)&(ymax>ymin),intersection,0)
  intersection3=tf.where(intersection2>0,intersection2,0)

  union=gt_box_size[tf.newaxis,:]+anchor_box_size-intersection3
  iou=intersection3/union

  #print(iou.shape)

  # 추가로 해야 할 부분
  # 도출된 iou값을 기준으로 index를 알아와서 positive와 negative 구분하기.
  positive_index=tf.where(iou>=0.7)
  iou_positive=tf.where(iou>=0.7)
  negative_index=tf.where(iou<0.3)
  iou_negative=tf.where(iou<0.3)
  positive_pos=tf.where(iou>=0.7,1,0)
  negative_pos=tf.where(iou<0.3,1,0)
  unuse_pos=tf.where(tf.logical_and(iou<0.7,iou>=0.3),1,0)

  nop=tf.reduce_sum(positive_pos)


  positive_ind=tf.stack([(positive_index[:,0]//9)//31,(positive_index[:,0]//9)%31,positive_index[:,0]%9,iou_positive[:,1]],axis=1)
  negative_ind=tf.stack([(negative_index[:,0]//9)//31,(negative_index[:,0]//9)%31,negative_index[:,0]%9,iou_negative[:,1]],axis=1)

  #print(positive_index.shape,negative_ind.shape)

  pdata=tf.zeros([0,4])
  pindex=tf.ones([0,4],dtype=tf.int64)

  if tf.math.less_equal(nop,tf.constant(128)):
    if tf.math.not_equal(nop,tf.constant(0)):
      pindex=tf.random.shuffle(positive_ind,name='positive_shuffle')
      #print(pindex.shape)
      pdata=tf.gather_nd(anchor,indices=[tf.stack([pindex[:,0],pindex[:,1],pindex[:,2]],axis=1)])
      pdata=tf.reshape(pdata,(-1,4))
      #number of positive
      nop=tf.shape(positive_index)[0]
    else:
      p_flag=1
  else:
    pindex=tf.random.shuffle(positive_ind,name='positive_shuffle')[:128]
    pdata=tf.gather_nd(anchor,indices=[tf.stack([pindex[:,0],pindex[:,1],pindex[:,2]],axis=1)])
    pdata=tf.reshape(pdata,(-1,4))
    nop=tf.constant(128)

  #number of negative
  non=tf.subtract(tf.constant(256),nop)
  nindex=tf.random.shuffle(negative_ind,name='negative_shuffle')[:non]
  ndata=tf.gather_nd(anchor,indices=[tf.stack([nindex[:,0],nindex[:,1],nindex[:,2]],axis=1)])
  ndata=tf.reshape(ndata,(-1,4))

  label_list=tf.concat([tf.ones((nop,1)),tf.zeros((non,1))],axis=0)
  if p_flag!=1:
    anchor_list=tf.concat([pdata,ndata],axis=0)
    gt_list=tf.concat([pindex[:,3],nindex[:,3]],axis=0)
    batch_pos=tf.concat([pindex[:,:3],nindex[:,:3]],axis=0)
    tindex=tf.random.shuffle(tf.range(256),name='total_shuffle')

    batch_label=tf.gather(label_list,indices=tindex)
    batch_anchor=tf.gather(anchor_list,indices=tindex)
    gt_list=tf.gather(gt_list,indices=tindex)
    batch_gt=tf.gather(gt_box,indices=gt_list)
    batch_pos=tf.gather(batch_pos,indices=tindex)
  else:
    anchor_list=ndata
    gt_list=nindex[:,3]
    batch_pos=nindex[:,:3]
    tindex=tf.random.shuffle(tf.range(256),name='total_shuffle')
    batch_label=tf.gather(label_list,indices=tindex)
    batch_anchor=tf.gather(anchor_list,indices=tindex)
    gt_list=tf.gather(gt_list,indices=tindex)
    batch_gt=tf.gather(gt_box,indices=gt_list)
    batch_pos=tf.gather(batch_pos,indices=tindex)

  w_a=batch_anchor[:,3]-batch_anchor[:,1]
  h_a=batch_anchor[:,2]-batch_anchor[:,0]
  t_x_star=(batch_gt[:,1]-batch_anchor[:,1])/w_a
  t_y_star=(batch_gt[:,0]-batch_anchor[:,0])/h_a
  t_w_star=tf.math.log((batch_gt[:,3]-batch_gt[:,1])/w_a)
  t_h_star=tf.math.log((batch_gt[:,2]-batch_gt[:,0])/h_a)
  batch_reg_gt=tf.stack([t_x_star,t_y_star,t_w_star,t_h_star],axis=1)

  return image,batch_label,batch_anchor,batch_gt,gt_list,batch_pos,batch_reg_gt
  # 추출해야할 요소 : Anchor_box의 좌표정보와 대응되는 gt_box의 좌표정보. 
  # making_rpn_train에서 iou정보를 이용해서 마지막 차원에 몇번째 gt_box와 대응되는지 알아야함.


# RPN Network

base_model = VGG16(include_top=False, input_shape=(500, 500, 3))
feature_extractor = base_model.get_layer("block5_conv3")
initializers=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
output=Conv2D(512,(3,3),activation='relu',kernel_initializer=initializers,padding='same')(feature_extractor.output)
rpn_cls_output = Conv2D(9, (1, 1), activation="sigmoid",kernel_initializer=initializers, name="rpn_cls")(output)
rpn_reg_output = Conv2D(9 * 4, (1, 1), activation="linear",kernel_initializer=initializers, name="rpn_reg")(output)
rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
rpn_model.summary()



# Huber Loss
class loss_bbr(keras.losses.Loss):
  def __init__(self,threshold=1,**kwargs):
    self.threshold=threshold
    super().__init__(**kwargs)

  def call(self, y_true,y_pred):
    error=y_true-y_pred
    is_small_error=tf.abs(error)<self.threshold
    squared_loss=tf.square(error)/2
    abs_loss=self.threshold * tf.abs(error)-self.threshold**2/2
    return tf.where(is_small_error,squared_loss,abs_loss)

  def get_config(self):
    base_config=super().get_config()
    return {**base_config}



epoch=300
optimizer=tf.keras.optimizers.Adam()
anchor_box=make_anchor()
train_loss_list=[10]
valid_loss_list=[10]
best_valid_loss_index=0
revision_count=0
loss_cls=tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_bbr=loss_bbr()
valid_set=voc_valid2.take(5)

#image_batch=5

voc_train3=voc_train2.map(patch_batch)
voc_train4=voc_train3.batch(5).prefetch(5)

voc_valid3=voc_valid2.map(patch_batch)
voc_valid4=voc_valid3.batch(1).prefetch(1)

# train_rpn_model

for epo in range(1,epoch+1):
  print("Epoch {}/{}".format(epo,epoch))
  train_total_loss=tf.constant(0,dtype=tf.float32)
  valid_total_loss=tf.constant(0,dtype=tf.float32)

  for data in tqdm(voc_train4):
    #image,batch_label,batch_anchor,batch_gt,gt_list,batch_pos,batch_reg_gt
    with tf.GradientTape() as tape:
      pred_reg,pred_obj=rpn_model(data[0],training=True)    # pred_reg = (5,31,31,36) , pred_obj= (5,31,31,9)
      pred_reg=tf.reshape(pred_reg,(-1,31,31,9,4)) # (5,31,31,9,4)
      reg_pos=tf.where(tf.math.equal(tf.cast(data[1],dtype=tf.int64),tf.constant(0,dtype=tf.int64))) # (5,?,4)
      y_t=tf.gather_nd(data[6],indices=reg_pos[:,:2])
      y_p=tf.gather_nd(tf.gather_nd(pred_reg,indices=data[5],batch_dims=1),indices=reg_pos[:,:2])
      pred_obj=tf.gather_nd(pred_obj,indices=data[5],batch_dims=1)# pred_obj= (5,256)
      pred_obj=tf.expand_dims(pred_obj,2)  
      objectness_loss=loss_cls(data[1],pred_obj) # batch_label(objectness) :(5,256,1) vs pred_obj : (5,256,1) 
      bounding_box_loss=loss_bbr(y_t,y_p)
      train_sub_loss=tf.add_n([objectness_loss/256]+[(bounding_box_loss*4/961)])  
    
    gradients=tape.gradient(train_sub_loss,rpn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,rpn_model.trainable_variables))
    train_total_loss=tf.add(train_total_loss,train_sub_loss)
  train_loss_list.append(tf.reduce_sum(train_total_loss)/501)

  for data in tqdm(voc_valid4):
    #image,batch_label,batch_anchor,batch_gt,gt_list,batch_pos,batch_reg_gt
    pred_reg_valid,pred_obj_valid=rpn_model(data[0],training=False)    # pred_reg = (5,31,31,36) , pred_obj= (5,31,31,9)
    pred_reg_valid=tf.reshape(pred_reg_valid,(-1,31,31,9,4)) # (5,31,31,9,4)
    reg_pos_valid=tf.where(tf.math.equal(tf.cast(data[1],dtype=tf.int64),tf.constant(0,dtype=tf.int64))) # (5,?,4)
    y_t_valid=tf.gather_nd(data[6],indices=reg_pos_valid[:,:2])
    y_p_valid=tf.gather_nd(tf.gather_nd(pred_reg_valid,indices=data[5],batch_dims=1),indices=reg_pos_valid[:,:2])
    pred_obj_valid=tf.gather_nd(pred_obj_valid,indices=data[5],batch_dims=1)# pred_obj= (5,256)
    pred_obj_valid=tf.expand_dims(pred_obj_valid,2)  
    objectness_loss=loss_cls(data[1],pred_obj_valid) # batch_label(objectness) :(5,256,1) vs pred_obj : (5,256,1) 
    bounding_box_loss=loss_bbr(y_t,y_p)
    valid_sub_loss=tf.add_n([objectness_loss/256]+[(bounding_box_loss*4/961)])
    valid_total_loss=tf.add(valid_total_loss,valid_sub_loss)

  valid_loss_list.append(valid_total_loss/2510)


  if epo%10==1:
    for valid in valid_set:
      valid_reg,valid_cls=rpn_model(tf.expand_dims(valid["image"],axis=0),training=False)
      print("GT Results")
      vision_valid(valid["image"],valid["bbox"])
      print("Model Results")
      v_pind=making_valid_positive(anchor_box,valid["bbox"])
      if v_pind.shape[0]!=0:
        v_pdata=tf.gather_nd(anchor_box,indices=tf.unstack(v_pind[:,:3]))
        v_cls=tf.gather_nd(tf.reshape(valid_cls,(31,31,9,1)),indices=tf.unstack(v_pind[:,:3]))
        proposed_box=tf.image.non_max_suppression(v_pdata,tf.reshape(v_cls,(-1)),5,iou_threshold=1.0)
        v_pdata = tf.gather(v_pdata, proposed_box)
        vision_valid(valid["image"],v_pdata)
      
  if valid_loss_list[best_valid_loss_index]>valid_loss_list[epo]:
    best_valid_loss_index=epo
    revision_count=0
    weight=rpn_model.get_weights()
  else:
    revision_count=revision_count+1
  
  if revision_count>=5:
    break
  print("Train_Loss = {}, Valid_Loss={}, revision_count = {}".format(train_loss_list[epo],valid_loss_list[epo],revision_count))



