import tensorflow as tf
import tensorflow_datasets as tfds

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


def data_loader(info=0):
    if info==1:
        dataset,info=tfds.load("voc",with_info=True,split=["test","train+validation[0%:95%]","validation[95%:]"])
        print(info.features['labels'].names)
    else:
        dataset,info=tfds.load("voc",with_info=True,split=["test","train+validation[0%:95%]","validation[95%:]"])

    voc_train,voc_test,voc_valid=dataset[1],dataset[0],dataset[2]
    voc_train2=voc_train.map(lambda feature: random_augmentation(feature))
    voc_test2=voc_test.map(lambda feature: data_preprocess(feature))
    voc_valid2=voc_valid.map(lambda feature: data_preprocess(feature))

    return voc_train2,voc_test2,voc_valid2


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
    pos=tf.gather(pos,indices=valid_ind)
    
    ind=tf.concat([tf.expand_dims(valid_ind,axis=1),pos],axis=1)
    
    pos2=tf.gather_nd(iou,indices=ind,batch_dims=0)
    survive_ind=tf.where(pos2>0.3)
    ind2=tf.gather_nd(ind,indices=survive_ind)
    iou2=tf.gather(iou,indices=valid_ind)
    
    negative_index=tf.where(tf.reduce_max(iou2,axis=1)<0.3)
    negative_index=tf.concat([negative_index,41*tf.ones((tf.shape(negative_index)[0],1),dtype=tf.int64)],axis=1)

    
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


def inverse_trans(anchor_box,pred_reg):
    
    '''
    inverse transform rpn offset
    '''
    
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


def generate_coord(proposed,pred_reg):
    
    '''
    inverse transform frcn offset
    '''
    
    w_a=tf.clip_by_value(proposed[:,:,3]-proposed[:,:,1],1e-2,1)
    h_a=tf.clip_by_value(proposed[:,:,2]-proposed[:,:,0],1e-2,1) 
    x=pred_reg[:,:,0]*w_a*0.1+proposed[:,:,1]
    y=pred_reg[:,:,1]*h_a*0.1+proposed[:,:,0]
    w=tf.math.exp(pred_reg[:,:,2]*0.2)*w_a
    h=tf.math.exp(pred_reg[:,:,3]*0.2)*h_a
    x_max=x+w
    y_max=y+h
    y=tf.clip_by_value(y,0,1)
    x=tf.clip_by_value(x,0,1)
    y_max=tf.clip_by_value(y_max,0,1)
    x_max=tf.clip_by_value(x_max,0,1)
    pred_value=tf.stack([y,x,y_max,x_max],axis=2)
    return pred_value

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
    p_iou=tf.where(iou5<0.5,20,plabel)
    p_iou=tf.where(iou5>=0.5,plabel,p_iou)
    
    gt_mask=tf.where(p_iou!=20,1,0)
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
            positive_index=tf.where(i!=20)
            positive_pos=tf.where(i!=20,1,0)
            negative_index=tf.where(i==20)
            
            nop=tf.clip_by_value(tf.reduce_sum(positive_pos),0,32)    
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
        positive_index=tf.where(p_iou!=20)
        positive_pos=tf.where(p_iou!=20,1,0)
        negative_index=tf.where(p_iou==20)
        
        nop=tf.clip_by_value(tf.reduce_sum(positive_pos),0,32)    
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