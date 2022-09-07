import subprocess
import sys

try:
    import neptune.new as neptune
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "neptune-client"])
    import neptune.new as neptune

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
from PIL import Image

run = neptune.init(
    project="sungsu/Faster-R-CNN",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTJmMGZiOC1jYzc0LTRkNTYtYWU1YS1jMGI0YmNmZDU4ZjgifQ==",
)

dataset,info=tfds.load("voc",with_info=True,split=["test","train+validation[0%:95%]","validation[95%:]"])
info.features['labels'].names

voc_train,voc_test,voc_valid=dataset[1],dataset[0],dataset[2]

def data_preprocess(feature):
    img=feature['image']
    label=feature['objects']['label']
    bbox=feature['objects']['bbox']
    paddings1 = [[0, 42-tf.shape(bbox)[0]], [0, 0]]
    bbox = tf.pad(bbox, paddings1, 'CONSTANT', constant_values=0)
    bbox=tf.reshape(bbox,(42,4))
    paddings2 = [[0,42-tf.shape(label)[0]]]
    label = tf.pad(label, paddings2, 'CONSTANT', constant_values=0)
    label= tf.reshape(label,(42,))
    image=tf.image.resize(img,[500,500])

    return {"image":image,"bbox":bbox,"label":label}



def augmentation_data(img,label,bbox):

    image=tf.image.flip_left_right(img)
    xmin=tf.reshape((1-bbox[:,3]),(-1,1))
    xmax=tf.reshape((1-bbox[:,1]),(-1,1))
    ymin=tf.reshape(bbox[:,0],(-1,1))
    ymax=tf.reshape(bbox[:,2],(-1,1))
    bbox_aug=tf.concat([ymin,xmin,ymax,xmax],axis=1)

    return [image,label,bbox_aug]


def random_augmentation(feature):
    img=feature['image']
    label=feature['objects']['label']
    bbox=feature['objects']['bbox']
    image=tf.image.resize(img,[500,500])
    num=tf.random.uniform([])
    if num>0.5:
        image,label,bbox=augmentation_data(image,label,bbox)    
    paddings1 = [[0, 42-tf.shape(bbox)[0]], [0, 0]]
    bbox = tf.pad(bbox, paddings1, 'CONSTANT', constant_values=0)
    bbox=tf.reshape(bbox,(42,4))
    paddings2 = [[0,42-tf.shape(label)[0]]]
    label = tf.pad(label, paddings2, 'CONSTANT', constant_values=0)
    label= tf.reshape(label,(42,))

    return {"image":image,"bbox":bbox,"label":label}

voc_train2=voc_train.map(lambda feature: random_augmentation(feature))

#voc_train2=voc_train.map(lambda feature: data_preprocess(feature))

voc_test2=voc_test.map(lambda feature: data_preprocess(feature))
voc_valid2=voc_valid.map(lambda feature: data_preprocess(feature))


# Utils

# make_anchor : generating anchor box 
# Paper propose : {scale : [128,256,512], ratio : {0.5, 1, 2}} 
# anchor shape : {31,31,9,4}

def make_anchor():
    width,height=tf.constant(31.0),tf.constant(31.0)
    x=tf.range(width) # width는 특성맵의 width
    y=tf.range(height) # height는 특성맵의 height
    X,Y=tf.meshgrid(x,y)
    center_x=tf.math.add(tf.cast(X,tf.float64),0.5)/31
    center_y=tf.math.add(tf.cast(Y,tf.float64),0.5)/31
    
    scale=tf.constant([128,256,512],dtype=tf.float64)
    ratio1=tf.constant([0.5,1,2],dtype=tf.float64)
    anchor_wh=[]
    
    
    for i in scale:
        for r1 in ratio1:
            w=tf.sqrt(tf.divide(tf.pow(tf.divide(i,500),2),r1)) # width는 원래 input image의 width : 500 가정
            h=tf.multiply(w,r1) # height는 원래 input image의 height : 500 가정
            anchor_wh.append([w/2,h/2])

    for i in tf.range(tf.shape(anchor_wh)[0]):
        xmin=tf.clip_by_value(center_x-anchor_wh[i][0], 0, 1)
        xmax=tf.clip_by_value(center_x+anchor_wh[i][0], 0, 1)
        ymin=tf.clip_by_value(center_y-anchor_wh[i][1], 0, 1)
        ymax=tf.clip_by_value(center_y+anchor_wh[i][1], 0, 1)
        if tf.equal(i,tf.constant(0)):
            anchor_box=tf.stack([ymin,xmin,ymax,xmax],axis=2)
            anchor_box=tf.expand_dims(anchor_box,axis=3)
        else:
            temp=tf.stack([ymin,xmin,ymax,xmax],axis=2)
            temp=tf.expand_dims(temp,axis=3)
            anchor_box=tf.concat([anchor_box,temp],axis=3)
    
    anchor_box=tf.transpose(anchor_box,[0,1,3,2])
    anchor_box=tf.cast(anchor_box,dtype=tf.float32)
    return anchor_box


# vision_valid : print image and bbox
def vision_valid(image,gt_box,visable=0):
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
        run["outputs/rpn_valid"].log(neptune.types.File.as_image(img))


def making_valid_positive(anchor_box,gt_box,iou_threshold):
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
    positive_index=tf.where(iou>=iou_threshold)
    iou_positive=tf.where(iou>=iou_threshold)


    positive_ind=tf.stack([(positive_index[:,0]//9)//31,(positive_index[:,0]//9)%31,positive_index[:,0]%9,iou_positive[:,1]],axis=1)
    print(positive_ind.shape)
    return positive_ind


# patch_batch : data preprocess for rpn model
# usage : tfds map function 

def patch_batch(data,anchor_box):
    image=data['image']
    gt_box=data['bbox']
    label=data['label']

    anchor_box=anchor_box
    anchor=anchor_box
    anchor_box3=tf.reshape(anchor_box,[31*31*9,4])

    valid_ind=tf.where(tf.reduce_sum(tf.where(((anchor_box3>0)&(anchor_box3<1)),1,0),axis=1)==4)
    valid_ind=tf.squeeze(valid_ind,axis=1)

    gt_box_size=(gt_box[:,2]-gt_box[:,0])*(gt_box[:,3]-gt_box[:,1])
    anchor_box2=tf.reshape(anchor_box,[31*31*9,1,4])
    anchor_box_size=(anchor_box2[:,:,2]-anchor_box2[:,:,0])*(anchor_box2[:,:,3]-anchor_box2[:,:,1])
      
    xmin=tf.math.maximum(anchor_box2[:,:,1],gt_box[:,1])
    ymin=tf.math.maximum(anchor_box2[:,:,0],gt_box[:,0])
    xmax=tf.math.minimum(anchor_box2[:,:,3],gt_box[:,3])
    ymax=tf.math.minimum(anchor_box2[:,:,2],gt_box[:,2])

    intersection=(xmax-xmin)*(ymax-ymin)
    intersection2=tf.where((xmax>xmin)&(ymax>ymin),intersection,0)
    intersection3=tf.where(intersection2>0,intersection2,0)

    union=gt_box_size[tf.newaxis,:]+anchor_box_size-intersection3
    iou=intersection3/union

    # 추가로 해야 할 부분
    # 도출된 iou값을 기준으로 index를 알아와서 positive와 negative 구분하기.
    pos=tf.expand_dims(tf.math.argmax(iou,axis=1),axis=1)
    #ind_list=tf.expand_dims(tf.range(0,8649,dtype=tf.int64),axis=1)
    pos=tf.gather(pos,indices=valid_ind)
    
    ind=tf.concat([tf.expand_dims(valid_ind,axis=1),pos],axis=1)
    
    pos2=tf.gather_nd(iou,indices=ind,batch_dims=0)
    survive_ind=tf.where(pos2>0.3)
    ind2=tf.gather_nd(ind,indices=survive_ind)
    #tf.where(tf.gather_nd(iou,indices=ind2)>0.7)
    
    #positive_index=tf.where(iou>=0.7)
    #iou_positive=tf.where(iou>=0.7)
    iou2=tf.gather(iou,indices=valid_ind)
    
    negative_index=tf.where(tf.reduce_max(iou2,axis=1)<0.3)
    negative_index=tf.concat([negative_index,41*tf.ones((tf.shape(negative_index)[0],1),dtype=tf.int64)],axis=1)
    #positive_pos=tf.where(iou>=0.7,1,0)
    #negative_pos=tf.where(iou<0.3,1,0)
    #unuse_pos=tf.where(tf.logical_and(iou<0.7,iou>=0.3),1,0)
    
    #nop=tf.reduce_sum(positive_pos)    
    
    nop=tf.shape(ind2)[0]
    nop2=tf.clip_by_value(nop,tf.constant(0),tf.constant(128))

    positive_ind=tf.stack([(ind2[:,0]//9)//31,(ind2[:,0]//9)%31,ind2[:,0]%9,ind2[:,1]],axis=1)
    negative_ind=tf.stack([(negative_index[:,0]//9)//31,(negative_index[:,0]//9)%31,negative_index[:,0]%9,negative_index[:,1]],axis=1)

    pindex=tf.random.shuffle(positive_ind,name='positive_shuffle')[:nop2]
    pdata=tf.gather_nd(anchor,indices=[tf.stack([pindex[:,0],pindex[:,1],pindex[:,2]],axis=1)])
    pdata=tf.reshape(pdata,(-1,4))

    #number of negative
    non=tf.subtract(tf.constant(256),nop2)
    nindex=tf.random.shuffle(negative_ind,name='negative_shuffle')[:non]
    ndata=tf.gather_nd(anchor,indices=[tf.stack([nindex[:,0],nindex[:,1],nindex[:,2]],axis=1)])
    ndata=tf.reshape(ndata,(-1,4))

    label_list=tf.concat([tf.ones((nop2,1)),tf.zeros((non,1))],axis=0)
    anchor_list=tf.concat([pdata,ndata],axis=0)
    gt_list=tf.concat([pindex[:,3],nindex[:,3]],axis=0)
    batch_pos=tf.concat([pindex[:,:3],nindex[:,:3]],axis=0)
    tindex=tf.random.shuffle(tf.range(256),name='total_shuffle')

    batch_label=tf.gather(label_list,indices=tindex)
    batch_anchor=tf.gather(anchor_list,indices=tindex)
    gt_list=tf.gather(gt_list,indices=tindex)
    batch_gt=tf.gather(gt_box,indices=gt_list)
    batch_pos=tf.gather(batch_pos,indices=tindex)

    # 상관없는 부분
    w_a=batch_anchor[:,3]-batch_anchor[:,1]
    h_a=batch_anchor[:,2]-batch_anchor[:,0]
    t_x_star=(batch_gt[:,1]-batch_anchor[:,1])/w_a/0.1
    t_y_star=(batch_gt[:,0]-batch_anchor[:,0])/h_a/0.1
    t_w_star=tf.math.log((batch_gt[:,3]-batch_gt[:,1])/w_a)/0.2
    t_h_star=tf.math.log((batch_gt[:,2]-batch_gt[:,0])/h_a)/0.2
    batch_reg_gt=tf.stack([t_x_star,t_y_star,t_w_star,t_h_star],axis=1)

    return image,batch_label,batch_anchor,batch_gt,gt_list,batch_pos,batch_reg_gt,label,gt_box
    # 추출해야할 요소 : Anchor_box의 좌표정보와 대응되는 gt_box의 좌표정보. 
    # making_rpn_train에서 iou정보를 이용해서 마지막 차원에 몇번째 gt_box와 대응되는지 알아야함.


def inverse_trans(anchor_box,pred_reg):
    pred_reg2=tf.reshape(pred_reg,(-1,31,31,9,4))
    # offset을 원래 ymin,xmin,ymax,xmax로 변환
    cal_anc=tf.expand_dims(anchor_box,axis=0)
    w_a=cal_anc[:,:,:,:,3]-cal_anc[:,:,:,:,1]
    h_a=cal_anc[:,:,:,:,2]-cal_anc[:,:,:,:,0]  
    x=pred_reg2[:,:,:,:,0]*w_a*0.1+cal_anc[:,:,:,:,1]
    y=pred_reg2[:,:,:,:,1]*h_a*0.1+cal_anc[:,:,:,:,0]
    w=tf.math.exp(pred_reg2[:,:,:,:,2]*0.2)*w_a
    h=tf.math.exp(pred_reg2[:,:,:,:,3]*0.2)*h_a
    x_max=x+w
    y_max=y+h
    y=tf.clip_by_value(y,0,1)
    x=tf.clip_by_value(x,0,1)
    y_max=tf.clip_by_value(y_max,0,1)
    x_max=tf.clip_by_value(x_max,0,1)
    pred_value=tf.stack([y,x,y_max,x_max],axis=4)
    return pred_value

def valid_result(valid,iou=0.5,max_n=300,visable=0):
    valid_reg,valid_cls=rpn_model(tf.expand_dims(valid["image"],axis=0),training=False)
    print("GT Results")
    vision_valid(valid["image"],valid["bbox"])
    print("Model Results")
    valid_reg2=inverse_trans(anchor_box,valid_reg)
    valid_reg3=tf.reshape(valid_reg2,(31,31,9,4))
    v_pind=making_valid_positive(valid_reg3,valid["bbox"],iou)
    if v_pind.shape[0]!=0:
        v_pdata=tf.gather_nd(valid_reg3,indices=tf.unstack(v_pind[:,:3]))
        # anchor_box.shape vs valid_reg.shape
        v_cls=tf.gather_nd(tf.reshape(valid_cls,(31,31,9,1)),indices=tf.unstack(v_pind[:,:3]))
        proposed_box=tf.image.non_max_suppression(v_pdata,tf.reshape(v_cls,(-1)),max_n,iou_threshold=0.5)
        v_pdata = tf.gather(v_pdata, proposed_box)
        vision_valid(valid["image"],v_pdata,visable)

# RPN Network Performance test
def valid_result2(valid,iou=0.3,max_n=300,visable=0):
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
    vision_valid(valid["image"],v_pdata,visable)

def generate_coord(proposed,pred_reg):
    pred_reg2=tf.reshape(pred_reg,(-1,1500,21,4))
    # offset을 원래 ymin,xmin,ymax,xmax로 변환
    cal_anc=tf.expand_dims(proposed,axis=2)
    w_a=cal_anc[:,:,:,3]-cal_anc[:,:,:,1]
    h_a=cal_anc[:,:,:,2]-cal_anc[:,:,:,0]  
    x=pred_reg2[:,:,:,0]*w_a*0.1+cal_anc[:,:,:,1]
    y=pred_reg2[:,:,:,1]*h_a*0.1+cal_anc[:,:,:,0]
    w=tf.math.exp(pred_reg2[:,:,:,2]*0.2)*w_a
    h=tf.math.exp(pred_reg2[:,:,:,3]*0.2)*h_a
    x_max=x+w
    y_max=y+h
    y=tf.clip_by_value(y,0,1)
    x=tf.clip_by_value(x,0,1)
    y_max=tf.clip_by_value(y_max,0,1)
    x_max=tf.clip_by_value(x_max,0,1)
    pred_value=tf.stack([y,x,y_max,x_max],axis=3)
    return pred_value

# Huber Loss
class Loss_bbr(tf.keras.losses.Loss):
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



# RPN Network

def construct_rpn(flag1=0):
    base_model = VGG16(include_top=False, input_shape=(500, 500, 3))
    feature_extractor = base_model.get_layer("block5_conv3")
    initializers=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    output=Conv2D(512,(3,3),activation='relu',kernel_initializer=initializers,padding='same')(feature_extractor.output)
    rpn_cls_output = Conv2D(9, (1, 1), activation="sigmoid",kernel_initializer=initializers, name="rpn_cls")(output)
    rpn_reg_output = Conv2D(9 * 4, (1, 1), activation="linear",kernel_initializer=initializers, name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[feature_extractor.output,rpn_reg_output, rpn_cls_output],name='RPN_Net')
    if flag1==1:
        rpn_model.load_weights("./model/rpn_FAS-73.h5")
    return rpn_model

## Implement NMS + ROI 풀링 ##

#pred_reg,pred_obj
#print(pred_reg.shape , pred_obj.shape)

## FAST_RCNN Layer
def construct_frcn(flag1=0):
    inputs=tf.keras.Input((1500,14,14,512),name='crop_image_interpolation')
    layer1=tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2,2)),name='ROI_Pool')(inputs)
    layer2=tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(),name='Flatten')(layer1)
    layer3=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4096,activation='relu'),name='fc1')(layer2)
    layer4=tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5),name='Dropout1')(layer3)
    layer5=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4096,activation='relu'),name='fc2')(layer4)
    layer6=tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5),name='Dropout2')(layer5)
    cls_layer=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(21,activation='softmax',name='classifier'))(layer6)
    reg_layer=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(21*4,activation='linear',name='bbox_correction'))(layer6)
    frcn_model=Model(inputs=inputs,outputs=[reg_layer,cls_layer],name='Faster_RCNN_Model')
    if flag1==1:
        frcn_model.load_weights("./model/frcn_FAS-57.h5")
    return frcn_model

def process_fmap(fmap,score,coord,anchor_box):
    pred_obj=tf.reshape(score,(tf.shape(fmap)[0],-1))
    a,b=tf.math.top_k(pred_obj,k=6000)    
    candidate=tf.stack([(b//9)//31,(b//9)%31,b%9],axis=2)
    pred_value=inverse_trans(anchor_box,coord)
    candidate_coord=tf.gather_nd(pred_value,indices=candidate,batch_dims=1)
    adjust_coord=tf.expand_dims(candidate_coord,axis=2)
    conf_score=tf.expand_dims(a,axis=2)
    proposed,_,_,_=tf.image.combined_non_max_suppression(adjust_coord,conf_score,1500,1500,iou_threshold=0.7)
    proposed2=tf.reshape(proposed,(-1,4)) 
    box_indices = tf.repeat(tf.range(tf.shape(fmap)[0]),tf.repeat(tf.constant(1500),tf.shape(fmap)[0]))
    crop_fmap=tf.image.crop_and_resize(fmap,proposed2,box_indices,(14,14))
    crop_fmap=tf.reshape(crop_fmap,(-1,1500,14,14,512))
    
    return crop_fmap,proposed

'''
class full_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.rpn_model=construct_rpn()
        self.frcn_model=construct_frcn()
        self.anchor_box=make_anchor()
        
        
    def call(self,x,train=False):
        
        fmap,reg_offset,object_score=self.rpn_model(x,training=train)
        crop_fmap=tf.stop_gradient(process_fmap(fmap,object_score,reg_offset,self.anchor_box))
        output=self.frcn_model(crop_fmap,training=train)
    
        return output
'''

'''
for data in voc_train4:
    fmap,pred_reg,pred_obj=rpn_model(data[0],training=False)    # pred_reg = (5,31,31,36) , pred_obj= (5,31,31,9)
    pred_reg=tf.reshape(pred_reg,(-1,31,31,9,4)) # (5,31,31,9,4)
    gt_box=data[8]
    label=data[7]
    break

'''

                
def making_frcnn_input(gt_box,label,fmap,pred_reg,pred_obj):
    
    pred_obj2=tf.reshape(pred_obj,(tf.shape(fmap)[0],-1))
    a,b=tf.math.top_k(pred_obj2,k=6000)    
    candidate=tf.stack([(b//9)//31,(b//9)%31,b%9],axis=2)
    pred_value=inverse_trans(anchor_box,pred_reg)
    candidate_coord=tf.gather_nd(pred_value,indices=candidate,batch_dims=1)
    adjust_coord=tf.expand_dims(candidate_coord,axis=2)
    conf_score=tf.expand_dims(a,axis=2)
    proposed,_,_,_=tf.image.combined_non_max_suppression(adjust_coord,conf_score,1500,1500,iou_threshold=0.7)
    proposed2=tf.reshape(proposed,(-1,4))    
    box_indices = tf.repeat(tf.range(tf.shape(fmap)[0]),tf.repeat(tf.constant(1500),tf.shape(fmap)[0]))
    crop_fmap=tf.image.crop_and_resize(fmap,proposed2,box_indices,(14,14))
    crop_fmap=tf.reshape(crop_fmap,(-1,1500,14,14,512))

    gt_box=tf.reshape(gt_box,(-1,1,42,4))
    gt_box_size=(gt_box[:,:,:,2]-gt_box[:,:,:,0])*(gt_box[:,:,:,3]-gt_box[:,:,:,1])

    ac=tf.expand_dims(proposed,axis=2)
    ac_size=(ac[:,:,:,2]-ac[:,:,:,0])*(ac[:,:,:,3]-ac[:,:,:,1])

    xmin=tf.math.maximum(ac[:,:,:,1],gt_box[:,:,:,1])
    ymin=tf.math.maximum(ac[:,:,:,0],gt_box[:,:,:,0])
    xmax=tf.math.minimum(ac[:,:,:,3],gt_box[:,:,:,3])
    ymax=tf.math.minimum(ac[:,:,:,2],gt_box[:,:,:,2])

    intersection=(xmax-xmin)*(ymax-ymin)
    intersection2=tf.where((xmax>xmin)&(ymax>ymin),intersection,0)
    intersection3=tf.where(intersection2>0,intersection2,1e-5)

    union=gt_box_size+ac_size-intersection3
    iou=intersection3/union
    #iou2=tf.where(iou>=0.5,iou,0)
    iou4=tf.where(iou>=0.1,iou,0)
    iou4=tf.clip_by_value(iou4,0,1)

    #iou3=tf.gather_nd(iou2,indices=tf.expand_dims(tf.math.argmax(iou2,axis=2),axis=2),batch_dims=2)
    plabel=tf.gather_nd(label,indices=tf.expand_dims(tf.math.argmax(iou4,axis=2),axis=2),batch_dims=1)

    iou5=tf.gather_nd(iou4,indices=tf.expand_dims(tf.math.argmax(iou4,axis=2),axis=2),batch_dims=2)
    #p_iou=tf.where(iou3>=0.5,plabel,-1)
    #p_iou=tf.where((iou5>=0.1)&(iou5<0.5),-1,plabel)
    p_iou=tf.where(iou5<0.5,-1,plabel)
    p_iou=tf.where(iou5>=0.5,plabel,p_iou)
    
    gt_mask=tf.where(p_iou!=-1,1,0)
    gt_label=tf.one_hot(p_iou,depth=21)
    gt_box2=tf.reshape(gt_box,(-1,42,4))

    p_coord=tf.gather_nd(gt_box2,indices=tf.expand_dims(tf.math.argmax(iou4,axis=2),axis=2),batch_dims=1)

    # Transform gtbox coord to offset
    w_a=tf.clip_by_value(proposed[:,:,3]-proposed[:,:,1],1e-2,1)
    h_a=tf.clip_by_value(proposed[:,:,2]-proposed[:,:,0],1e-2,1)
    t_x_star=(p_coord[:,:,1]-proposed[:,:,1])/w_a/0.1
    t_y_star=(p_coord[:,:,0]-proposed[:,:,0])/h_a/0.1
    t_w_star=tf.math.log((p_coord[:,:,3]-p_coord[:,:,1])/w_a)/0.2
    t_h_star=tf.math.log((p_coord[:,:,2]-p_coord[:,:,0])/h_a)/0.2

    gt_coord=tf.stack([t_x_star,t_y_star,t_w_star,t_h_star],axis=2)

    # RPN Train 과 비슷하게, positive 128, negative 128개 추출
    # P가 128개보다 작을 경우, N으로 채워넣음.
    # P인 경우만 Reg Loss 계산, 나머지 256개는 전부다 Classifier Loss 계산
    # 현재, BG인 경우와 FG인 경우 식별 가능한 변수 => 
    
    if tf.shape(crop_fmap)[0]==2:
        for n,i in enumerate(p_iou):
            i=tf.expand_dims(i,axis=0)
            positive_index=tf.where(i!=-1)
            positive_pos=tf.where(i!=-1,1,0)
            negative_pos=tf.where(i==-1,1,0)
            negative_index=tf.where(i==-1)
            
            print("positive Num : ",tf.reduce_sum(positive_pos),"Negative Num : ",tf.reduce_sum(negative_pos))
            
            nop=tf.clip_by_value(tf.reduce_sum(positive_pos),0,64)    
            non=tf.constant(128,dtype=tf.int32)-nop
        
            pindex=tf.random.shuffle(positive_index,name='positive_shuffle')[:nop]
            nindex=tf.random.shuffle(negative_index,name='negative_shuffle')[:non]
                        
            if tf.equal(n,tf.constant(0)):
                tindex=tf.concat([pindex,nindex],axis=0)
                tindex=tf.random.shuffle(tindex,name='total_shuffle')
                gt_label2=tf.expand_dims(tf.gather_nd(gt_label[0,tf.newaxis],indices=tindex),axis=0)
                gt_coord2=tf.expand_dims(tf.gather_nd(gt_coord[0,tf.newaxis],indices=tindex),axis=0)
                proposed2=tf.expand_dims(tf.gather_nd(proposed[0,tf.newaxis],indices=tindex),axis=0)
                gt_mask2=tf.expand_dims(tf.gather_nd(gt_mask[0,tf.newaxis],indices=tindex),axis=0)
            else:
                temp_tindex=tf.concat([pindex,nindex],axis=0)
                temp_tindex=tf.random.shuffle(temp_tindex,name='total_shuffle')
                temp_label2=tf.expand_dims(tf.gather_nd(gt_label[1,tf.newaxis],indices=temp_tindex),axis=0)
                temp_coord2=tf.expand_dims(tf.gather_nd(gt_coord[1,tf.newaxis],indices=temp_tindex),axis=0)
                temp_proposed2=tf.expand_dims(tf.gather_nd(proposed[1,tf.newaxis],indices=temp_tindex),axis=0)
                temp_gt_mask2=tf.expand_dims(tf.gather_nd(gt_mask[1,tf.newaxis],indices=temp_tindex),axis=0)
                
                gt_label2=tf.concat([gt_label2,temp_label2],axis=0)
                gt_coord2=tf.concat([gt_coord2,temp_coord2],axis=0)
                proposed2=tf.concat([proposed2,temp_proposed2],axis=0)    
                gt_mask2=tf.concat([gt_mask2,temp_gt_mask2],axis=0)
                tindex=tf.concat([tf.expand_dims(tindex[:,1],axis=0),tf.expand_dims(temp_tindex[:,1],axis=0)],axis=0)
    else:
        positive_index=tf.where(p_iou!=-1)
        positive_pos=tf.where(p_iou!=-1,1,0)
        negative_index=tf.where(p_iou==-1)
        
        nop=tf.clip_by_value(tf.reduce_sum(positive_pos),0,64)    
        non=tf.constant(128,dtype=tf.int32)-nop
        pindex=tf.random.shuffle(positive_index,name='positive_shuffle')[:nop]
        nindex=tf.random.shuffle(negative_index,name='negative_shuffle')[:non]
        
        tindex=tf.concat([pindex,nindex],axis=0)
        tindex=tf.random.shuffle(tindex,name='total_shuffle')
            
        gt_label2=tf.expand_dims(tf.gather_nd(gt_label,indices=tindex),axis=0)
        gt_coord2=tf.expand_dims(tf.gather_nd(gt_coord,indices=tindex),axis=0)
        proposed2=tf.expand_dims(tf.gather_nd(proposed,indices=tindex),axis=0)
        gt_mask2=tf.expand_dims(tf.gather_nd(gt_mask,indices=tindex),axis=0)
        tindex=tf.expand_dims(tindex[:,1],axis=0)

    '''
    output 
    
    crop_fmap : (B,1500,14,14,512)
    gt_label2 : (B,256,21)
    gt_mask2 : (B,256)
    gt_coord2 : (B,256,4)
    proposed2 : (B,256,4)
    tindex : (B,256)
    
    '''
    return crop_fmap,gt_label2,gt_mask2,gt_coord2,proposed,tindex

epoch=3000
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
rpn_train_loss_list=[1000]
rpn_valid_loss_list=[1000]
frcn_train_loss_list=[1000]
frcn_valid_loss_list=[1000]

best_valid_loss_index=0
revision_count=0
rpn_loss_cls=tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss_cls=tf.keras.losses.CategoricalCrossentropy()
loss_bbr=Loss_bbr()
anchor_box=make_anchor()
valid_set=voc_valid2.take(5).batch(1)
rpn_model=construct_rpn(flag1=1)
frcn_model=construct_frcn()


voc_train3=voc_train2.map(lambda x,y=anchor_box :patch_batch(x,y))
voc_train4=voc_train3.batch(2).prefetch(2)

voc_valid3=voc_valid2.map(lambda x,y=anchor_box :patch_batch(x,y))
voc_valid4=voc_valid3.batch(1).prefetch(1)

for epo in range(1,epoch+1):
    print("Epoch {}/{}".format(epo,epoch))
    
    rpn_train_total_loss=tf.constant(0,dtype=tf.float32)
    rpn_valid_total_loss=tf.constant(0,dtype=tf.float32)
    frcn_train_total_loss=tf.constant(0,dtype=tf.float32)
    frcn_valid_total_loss=tf.constant(0,dtype=tf.float32)

    for data in tqdm(voc_train4):
        #image,batch_label,batch_anchor,batch_gt,gt_list,batch_pos,batch_reg_gt
        fmap,pred_reg,pred_obj=rpn_model(data[0],training=False)    # pred_reg = (5,31,31,36) , pred_obj= (5,31,31,9)
        pred_reg=tf.reshape(pred_reg,(-1,31,31,9,4)) # (5,31,31,9,4)
        reg_pos=tf.where(tf.math.equal(tf.cast(data[1],dtype=tf.int64),tf.constant(1,dtype=tf.int64))) # (5,?,4)
        y_t=tf.gather_nd(data[6],indices=reg_pos[:,:2])
        y_p=tf.gather_nd(tf.gather_nd(pred_reg,indices=data[5],batch_dims=1),indices=reg_pos[:,:2])
        pred_obj2=tf.gather_nd(pred_obj,indices=data[5],batch_dims=1)# pred_obj= (5,256)
        pred_obj2=tf.expand_dims(pred_obj2,2)  
        rpn_objectness_loss=rpn_loss_cls(data[1],pred_obj2) # batch_label(objectness) :(5,256,1) vs pred_obj : (5,256,1) 
        rpn_bounding_box_loss=loss_bbr(y_t,y_p)
        #train_sub_loss=tf.add_n([objectness_loss/256]+[(bounding_box_loss*3.75/961)])
        rpn_train_sub_loss=tf.add_n([2*rpn_objectness_loss]+[rpn_bounding_box_loss])
    
        fmap,label,gt_mask,gt_coord,_,tindex=making_frcnn_input(data[8],data[7],fmap,pred_reg,pred_obj)
        
        cal_pos=tf.math.argmax(label,axis=2)
        cal_pos=tf.expand_dims(cal_pos,axis=2)
        tindex=tf.expand_dims(tindex,axis=2)
        pos_ind=tf.where(gt_mask!=0)
        gt_reg=tf.clip_by_value(gt_coord,-100,100)
        
        with tf.GradientTape() as tape2:
            pred_reg,pred_obj=frcn_model(fmap,training=True)   # pred_reg = (B,2000,84) , pred_obj= (B,2000,21)
            pred_reg=tf.reshape(pred_reg,(-1,1500,21,4))
            reg1=tf.gather_nd(pred_reg,indices=tindex,batch_dims=1)
            reg2=tf.gather_nd(reg1,indices=cal_pos,batch_dims=2)
            
            pred_reg=tf.gather_nd(reg2,indices=pos_ind)
            gt_reg=tf.gather_nd(gt_reg,indices=pos_ind)
            obj1=tf.gather_nd(pred_obj,indices=tindex,batch_dims=1)
            
            frcn_objectness_loss=loss_cls(label,obj1)  
            frcn_bounding_box_loss=loss_bbr(gt_reg,pred_reg)
            frcn_train_sub_loss=tf.add_n([2*frcn_objectness_loss]+[(frcn_bounding_box_loss)])
        
        
        print(frcn_objectness_loss,frcn_bounding_box_loss)
        gradients2=tape2.gradient(frcn_train_sub_loss,frcn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients2,frcn_model.trainable_variables))        
        #gradients=tape1.gradient(rpn_train_sub_loss,rpn_model.trainable_variables)
        #optimizer.apply_gradients(zip(gradients,rpn_model.trainable_variables))
        
        frcn_train_total_loss=tf.add(frcn_train_total_loss,frcn_train_sub_loss)
        rpn_train_total_loss=tf.add(rpn_train_total_loss,rpn_train_sub_loss)
        
        
        run["train/rpn_iter_loss"].log(rpn_train_sub_loss)
        run["train/rpn_obj_loss"].log(2*rpn_objectness_loss)
        run["train/rpn_reg_loss"].log(rpn_bounding_box_loss)
        run["train/frcn_iter_obj_loss"].log(2*frcn_objectness_loss)
        run["train/frcn_iter_reg_loss"].log(frcn_bounding_box_loss)
        run["train/frcn_iter_loss"].log(frcn_train_sub_loss)
        
    
    rpn_train_loss_list.append(tf.reduce_sum(rpn_train_total_loss)/2443) 
    run["train/rpn_epoch_loss"].log(tf.reduce_sum(rpn_train_total_loss)/2443)
    
    frcn_train_loss_list.append(tf.reduce_sum(frcn_train_total_loss)/2443)
    run["train/frcn_epoch_loss"].log(tf.reduce_sum(frcn_train_total_loss)/2443)
    
    
    for data in tqdm(voc_valid4):
        
        fmap,pred_reg,pred_obj=rpn_model(data[0],training=False)    # pred_reg = (5,31,31,36) , pred_obj= (5,31,31,9)
        pred_reg=tf.reshape(pred_reg,(-1,31,31,9,4)) # (5,31,31,9,4)
        reg_pos=tf.where(tf.math.equal(tf.cast(data[1],dtype=tf.int64),tf.constant(1,dtype=tf.int64))) # (5,?,4)
        y_t=tf.gather_nd(data[6],indices=reg_pos[:,:2])
        y_p=tf.gather_nd(tf.gather_nd(pred_reg,indices=data[5],batch_dims=1),indices=reg_pos[:,:2])
        pred_obj2=tf.gather_nd(pred_obj,indices=data[5],batch_dims=1)# pred_obj= (5,256)
        pred_obj2=tf.expand_dims(pred_obj2,2)  
        rpn_objectness_loss=rpn_loss_cls(data[1],pred_obj2) # batch_label(objectness) :(5,256,1) vs pred_obj : (5,256,1) 
        rpn_bounding_box_loss=loss_bbr(y_t,y_p)
        rpn_valid_sub_loss=tf.add_n([2*rpn_objectness_loss]+[rpn_bounding_box_loss])
        
        fmap,label,gt_mask,gt_coord,_,tindex=making_frcnn_input(data[8],data[7],fmap,pred_reg,pred_obj)
        
        cal_pos=tf.math.argmax(label,axis=2)
        cal_pos=tf.expand_dims(cal_pos,axis=2)
        tindex=tf.expand_dims(tindex,axis=2)
        pos_ind=tf.where(gt_mask!=0)
        gt_reg=tf.clip_by_value(gt_coord,-100,100)
        

        pred_reg,pred_obj=frcn_model(fmap,training=False)   # pred_reg = (B,2000,84) , pred_obj= (B,2000,21)
        pred_reg=tf.reshape(pred_reg,(-1,1500,21,4))
        reg1=tf.gather_nd(pred_reg,indices=tindex,batch_dims=1)
        reg2=tf.gather_nd(reg1,indices=cal_pos,batch_dims=2)
        
        pred_reg=tf.gather_nd(reg2,indices=pos_ind)
        gt_reg=tf.gather_nd(gt_reg,indices=pos_ind)
        obj1=tf.gather_nd(pred_obj,indices=tindex,batch_dims=1)
        
        frcn_objectness_loss=loss_cls(label,obj1) 
        frcn_bounding_box_loss=loss_bbr(gt_reg,pred_reg)
        frcn_valid_sub_loss=tf.add_n([2*frcn_objectness_loss]+[(frcn_bounding_box_loss)])  
        
        frcn_valid_total_loss=tf.add(frcn_valid_total_loss,frcn_valid_sub_loss)
        rpn_valid_total_loss=tf.add(rpn_valid_total_loss,rpn_valid_sub_loss)
        
        
        run["valid/rpn_iter_loss"].log(rpn_valid_sub_loss)
        run["valid/rpn_obj_loss"].log(2*rpn_objectness_loss)
        run["valid/rpn_reg_loss"].log(rpn_bounding_box_loss)
        run["valid/frcn_iter_obj_loss"].log(2*frcn_objectness_loss)
        run["valid/frcn_iter_reg_loss"].log(frcn_bounding_box_loss)
        run["valid/frcn_iter_loss"].log(frcn_valid_sub_loss)
        
    
    rpn_valid_loss_list.append(tf.reduce_sum(rpn_valid_total_loss)/126) 
    run["valid/rpn_epoch_loss"].log(tf.reduce_sum(rpn_valid_total_loss)/126)
    
    frcn_valid_loss_list.append(tf.reduce_sum(frcn_valid_total_loss)/126)
    run["valid/frcn_epoch_loss"].log(tf.reduce_sum(frcn_valid_total_loss)/126)
    
    
    if True:
        for valid in valid_set:
            fmap,pred_reg,pred_obj=rpn_model(valid['image'],training=False)    # pred_reg = (5,31,31,36) , pred_obj= (5,31,31,9)
            crop_fmap,proposed=process_fmap(fmap,pred_obj,pred_reg,anchor_box)

            pred_reg_valid,pred_obj_valid=frcn_model(crop_fmap,training=False)  
            result=generate_coord(proposed,pred_reg_valid)
            
            argindex=tf.math.argmax(pred_obj_valid,axis=2)
            bbox=tf.gather_nd(result,indices=tf.expand_dims(argindex,axis=2),batch_dims=2)
            pred_obj=tf.gather_nd(pred_obj_valid,indices=tf.expand_dims(argindex,axis=2),batch_dims=2)
            bgind=tf.where(argindex!=20)
            score=tf.gather_nd(pred_obj,indices=bgind)
            coord=tf.gather_nd(bbox,indices=bgind)
            proposed=tf.image.non_max_suppression_with_scores(coord,score,30,iou_threshold=0.7)
            fgind=tf.where(proposed[1]>=0.9)
            v_pdata = tf.gather(coord, proposed[0])
            v_pdata= tf.gather_nd(v_pdata,indices=fgind)        
            vision_valid(tf.reshape(valid['image'],(500,500,3)),v_pdata,visable=1)
        
    if frcn_valid_loss_list[best_valid_loss_index]>frcn_valid_loss_list[epo]:
        best_valid_loss_index=epo
        revision_count=0
        weight=frcn_model.get_weights()
        
    else:
        revision_count=revision_count+1
    
    if revision_count>=20:
        break
    print("Epoch = {}, Frcn_Train_Loss = {}, Frcn_Valid_Loss={} revision_count = {}".format(epo,frcn_train_loss_list[epo],frcn_valid_loss_list[epo],revision_count))


#rpn_model.set_weights(weight1)
frcn_model.set_weights(weight)
url=run.get_run_url().split('/')[-1]
#rpn_model.save_weights(f"./model/model_rpn_{url}.h5")
frcn_model.save_weights(f"./model/model_frcn_{url}.h5")
run.stop()





