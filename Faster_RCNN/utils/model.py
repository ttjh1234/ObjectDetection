import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
from .data import inverse_trans
from .anchor import make_anchor


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
        rpn_model.load_weights("./model/model_rpn_FAS-89.h5")
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
        frcn_model.load_weights("./model/model_frcn_FAS-89.h5")
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


class full_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.rpn_model=construct_rpn(flag1=1)
        self.frcn_model=construct_frcn(flag1=1)
        self.anchor_box=make_anchor()
        
        
    def call(self,x,train=False):
        
        fmap,reg_offset,object_score=self.rpn_model(x,training=train)
        crop_fmap=tf.stop_gradient(process_fmap(fmap,object_score,reg_offset,self.anchor_box))
        output=self.frcn_model(crop_fmap,training=train)
    
        return output
