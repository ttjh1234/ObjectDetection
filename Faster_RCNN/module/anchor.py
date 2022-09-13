import tensorflow as tf
import numpy as np

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