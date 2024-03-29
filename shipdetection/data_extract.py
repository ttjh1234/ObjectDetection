import os
import xml.etree.ElementTree as Et
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zipfile
import json
import re
import shutil
import random


os.getcwd()
os.chdir("C:/Users/UOS/Desktop/새론")

def unzip(source_file, dest_path):
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            if len(re.findall('[^가-힣a-zA-Z0-9_.{./}]+',member.filename))>0:
                member.filename = member.filename.encode("cp437").decode("euc-kr")
            if str.split(member.filename,'.')[-1]!="json":
                zf.extract(member,dest_path)

'''
Usage Example
#unzip("./[라벨]동해_묵호항_1구역_BOX.zip","./")
#unzip("./제주항_맑음_20201227_0848_0004.zip","../")
#unzip("./[라벨]서해_군산항_1구역_BOX.zip","./")
'''

def serialize_example(dic):
    image = dic["image"].tobytes()
    image_shape=dic["image_shape"].tobytes()
    bbox = dic["bbox"].tobytes()
    bbox_shape = dic["bbox_shape"].tobytes()
    filename = dic["filename"].encode('utf-8')

    feature_dict = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        "bbox": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
        "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        "image_shape": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_shape])),
        "bbox_shape": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_shape]))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example.SerializeToString()


def parse_func(example):
    '''
    fixed length padding     
    '''
    image = tf.io.decode_raw(example["image"], tf.float32)
    image_shape = tf.io.decode_raw(example["image_shape"], tf.int32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    bbox_shape = tf.io.decode_raw(example["bbox_shape"], tf.int32)
    image=tf.reshape(image,image_shape)
    bbox=tf.reshape(bbox,bbox_shape)
    paddings1 = [[0, 24-tf.shape(bbox)[0]], [0, 0]]
    bbox = tf.pad(bbox, paddings1, 'CONSTANT', constant_values=0)
        
    #filename=tf.py_function(decode_str,[example["filename"]],[tf.string])
    filename=example["filename"]

    return {'image':image,'bbox':bbox,'filename':filename}

def decode_filename(strtensor):
    strtensor=strtensor.numpy()
    filename=[str(i,'utf-8') for i in strtensor]
    return filename


def deserialize_example(serialized_string):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_shape": tf.io.FixedLenFeature([], tf.string),
        "bbox": tf.io.FixedLenFeature([],tf.string),
        "bbox_shape": tf.io.FixedLenFeature([],tf.string),
        "filename": tf.io.FixedLenFeature([],tf.string)
    }
    example = tf.io.parse_example(serialized_string, image_feature_description)
    dataset=parse_func(example)
            
    return dataset


def fetch_data(path):
    
    '''
    Function fetch_data 
    This is single specific path task
    General Version is fetch_data2 which is below that.
    '''
    
    obj_num=0
    image_path= "./[원천]남해_여수항_2구역_BOX/남해_여수항_2구역_BOX/"+path
    xml_path_ = "./[라벨]남해_여수항_2구역_BOX/남해_여수항_2구역_BOX/"+str.split(path,'.')[0]+".xml"
    
    xml =  open(xml_path_, mode = 'r', encoding="utf-8")
    xml_tree = Et.parse(xml) 
    xml_root = xml_tree.getroot()
    img_name=xml_root.find("filename").text
    objects = xml_root.findall("object")
    size = xml_root.find("size") 
    width = int(size.find("width").text)  
    height= int(size.find("height").text)
    image_shape=np.array([512,512,3])
    bndbox_list=tf.zeros([0,4],dtype=tf.float32)

    for i in objects:
        if i.find("category_id").text=='2':
            obj_num=obj_num+1
            bndbox=i.find("bndbox") # object 한 객체내에 bndbox 접근
            xmin=tf.cast(int(bndbox.find('xmin').text),dtype=tf.float32)/width # x최소 좌표 
            xmax=tf.cast(int(bndbox.find('xmax').text),dtype=tf.float32)/width # x최대 좌표
            ymin=tf.cast(int(bndbox.find('ymin').text),dtype=tf.float32)/height # y최소 좌표 
            ymax=tf.cast(int(bndbox.find('ymax').text),dtype=tf.float32)/height # y최대 좌표
            temp_box=tf.reshape(tf.stack([ymin,xmin,ymax,xmax]),(-1,4))
            bndbox_list=tf.concat([bndbox_list,temp_box],axis=0)

    if obj_num>=3:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2=tf.cast(img,dtype=tf.float64)
        img2=tf.image.resize(img2,(512,512))/255.0
        bbox_shape=np.array([obj_num,4])

        return {"filename":img_name,"image":img2.numpy(),"image_shape":image_shape,"bbox":bndbox_list.numpy(),"bbox_shape":bbox_shape}

    else:
        return None

'''
    # Fetch_data Usage
    # First Data Read
    # Second Data Parse and Match
    # Third Data Serialize 
'''


'''
pt="./[원천]남해_여수항_2구역_BOX/남해_여수항_2구역_BOX/"


with tf.io.TFRecordWriter("test.tfrecord") as f:
    for i in os.listdir(pt):
        subpath1=i+'/'    
        for j in os.listdir(pt+subpath1):
            subpath2=subpath1+j+'/'
            for k in os.listdir(pt+subpath2):
                data_path=subpath2+k
                dic=fetch_data(data_path)
                if dic:
                    result=serialize_example(dic)
                    f.write(result)
'''                   


# 여기서, 한글 파일명은 일단은 utf-8로 바뀌어서 나오는데, eager tensor에서는 무슨 이유인지는 몰라도
# 계속 바뀌지 않는 오류가 발생함. 따라서, 우선 나오도록하고, 추후에 str(filename,'utf-8) 로 바꾸어서 출력

path="E:/해상 객체 이미지/Training"
def filename_extract(path):
    
    '''
    Given Path, Extract Filename along Path
    except Seg.zip File
    Only Use Box.zip
    Extract Main Name except extension
    '''
    
    filelist=[]    
    allfile=os.listdir(path)
    rep=".*.zip"
    for i in allfile:
        fname=str.split(i,']')[-1]
        if not re.search(rep,fname):
            continue
        chunkfname=str.split(fname,'.zip')[0]
        if len(str.split(chunkfname,"BOX"))==1:
            continue
        
        chunkfname=str.split(chunkfname,"BOX")[0]+"BOX"
        if chunkfname not in filelist:
            filelist.append(chunkfname)
    
    return filelist

train_fname=filename_extract("E:/해상 객체 이미지/Training")
        
def find_file(path,file_list):
    
    '''
    Given Path & Filename(except extension) , Extract Filename(including Extension) 
    And Generate List which bind Image File & Annotation File    
    '''
    
    Filelist=[]
    for d in file_list:
        flist=[]
        for i in os.listdir(path):
            rep="[\[라벨|원천].*"+d+".+"
            temp=re.search(rep,i)
            if temp:
                if temp.group() not in flist:
                    flist.append(temp.group())
        Filelist.append(flist)
    return Filelist


file_name=find_file(path,train_fname)


os.getcwd()
os.chdir("C:/Users/UOS/Desktop/새론")

'''
Below that, Binded File to extract Image Path & Annotation Path
Because, There are Many-to-one Matching, So, later iterate Loop and open that File pair 
'''

origin_list=[]
label_list=[]
for t in file_name:
    subo_list=[]
    subl_list=[]
    for k in t:
        if re.search("원천",k):
            subo_list.append("./train/"+str.split(k,".zip")[0])
        if re.search("라벨",k):
            subl_list.append("./train/"+str.split(k,".zip")[0])
    origin_list.append(subo_list)
    label_list.append(subl_list)


'''
# Checking Result

# file_name
# origin_list
# label_list

'''


def fetch_data2(image_path,xml_path1,xml_path2):
    
    '''
    General Version of fetch_data
    Given image_path, Match Image File name & annotation File 
    but there are not standard path address.
    So split path to handle easily.  
    '''
    
    obj_num=0
    image_path= image_path
    xml_path_ = xml_path1+str.split(xml_path2,'.')[0]+".xml"
    
    try:
        xml =  open(xml_path_, mode = 'r', encoding="utf-8")
    except:
        return None
    
    xml_tree = Et.parse(xml) 
    xml_root = xml_tree.getroot()
    img_name=xml_root.find("filename").text
    objects = xml_root.findall("object")
    size = xml_root.find("size") 
    width = int(size.find("width").text)  
    height= int(size.find("height").text)
    image_shape=np.array([512,512,3])
    bndbox_list=tf.zeros([0,4],dtype=tf.float32)

    for i in objects:
        if i.find("category_id").text=='2':
            obj_num=obj_num+1
            bndbox=i.find("bndbox") # object 한 객체내에 bndbox 접근
            xmin=tf.cast(int(bndbox.find('xmin').text),dtype=tf.float32)/width # x최소 좌표 
            xmax=tf.cast(int(bndbox.find('xmax').text),dtype=tf.float32)/width # x최대 좌표
            ymin=tf.cast(int(bndbox.find('ymin').text),dtype=tf.float32)/height # y최소 좌표 
            ymax=tf.cast(int(bndbox.find('ymax').text),dtype=tf.float32)/height # y최대 좌표
            temp_box=tf.reshape(tf.stack([ymin,xmin,ymax,xmax]),(-1,4))
            bndbox_list=tf.concat([bndbox_list,temp_box],axis=0)

    if obj_num>=3:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2=tf.cast(img,dtype=tf.float64)
        img2=tf.image.resize(img2,(512,512))/255.0
        bbox_shape=np.array([obj_num,4])

        xml.close()
        return {"filename":img_name,"image":img2.numpy(),"image_shape":image_shape,"bbox":bndbox_list.numpy(),"bbox_shape":bbox_shape}

    else:
        xml.close()
        return None

'''
source = "E:/해상 객체 이미지/Training/"
with tf.io.TFRecordWriter("train.tfrecord") as f:
    for a,b,c in zip(file_name,origin_list,label_list):
        for d in a:
            if str.split(d,'.')[0] in os.listdir("./train"):
                continue
            else:
                shutil.copy2(source+d, "./train")
                unzip("./train/"+d,"./train/"+str.split(d,'.')[0])
                os.remove("./train/"+d)
        print("Unzip & Remove zip file in one port\n")
        c=c*len(b)
        print("Start Extract data in one port\n")
        for pt,pl in zip(b,c):
            xml_path1=pl+'/'+os.listdir(pl)[0]+'/'
            for t1 in os.listdir(pt):
                for i in os.listdir(pt+'/'+t1):
                    subpath1=t1+'/'+i+'/'    
                    for j in os.listdir(pt+'/'+subpath1):
                        subpath2=subpath1+j+'/'
                        rpath=os.listdir(pt+'/'+subpath2)
                        random.shuffle(rpath)
                        jump_var=0
                        for k in rpath:
                            if jump_var>0:
                                jump_var=jump_var-1
                                continue
                            data_path=subpath2+k
                            full_path=pt+'/'+data_path
                            fp="/"+str.join('/',str.split(full_path,'/')[4:])
                            dic=fetch_data2(full_path,xml_path1,fp)
                            if dic:
                                jump_var=2
                                result=serialize_example(dic)
                                f.write(result)
                                                
            shutil.rmtree(pt)
        shutil.rmtree(pl)
        print("End Extract data in one port\n")
'''


## validation 파일 추출
# train에 사용된 항구는 제외하여 사용.


os.getcwd()
os.chdir("C:/Users/UOS/Desktop/새론")

path="E:/해상 객체 이미지/Validation"
def filename_extract(path):
    filelist=[]    
    allfile=os.listdir(path)
    rep=".*.zip"
    for i in allfile:
        fname=str.split(i,']')[-1]
        if not re.search(rep,fname):
            continue
        chunkfname=str.split(fname,'.zip')[0]
        if len(str.split(chunkfname,"BOX"))==1:
            continue
        
        chunkfname=str.split(chunkfname,"BOX")[0]+"BOX"
        if chunkfname not in filelist:
            filelist.append(chunkfname)
    
    return filelist

valid_fname=filename_extract("E:/해상 객체 이미지/Validation")

def find_file(path,file_list):
    Filelist=[]
    for d in file_list:
        flist=[]
        for i in os.listdir(path):
            rep="[\[라벨|원천].*"+d+".+"
            temp=re.search(rep,i)
            if temp:
                if temp.group() not in flist:
                    flist.append(temp.group())
        Filelist.append(flist)
    return Filelist

valid_fname
train_fname

vf=set(valid_fname)
tf=set(train_fname)
vf2=vf.difference(tf)
vf3=list(vf2)


file_name_valid=find_file(path,vf3)


valid_origin_list=[]
valid_label_list=[]
for t in file_name_valid:
    subo_list=[]
    subl_list=[]
    for k in t:
        if re.search("원천",k):
            subo_list.append("./valid/"+str.split(k,".zip")[0])
        if re.search("라벨",k):
            subl_list.append("./valid/"+str.split(k,".zip")[0])
    valid_origin_list.append(subo_list)
    valid_label_list.append(subl_list)

file_name_valid[:7]
valid_origin_list[:7]
valid_label_list[:7]


source = "E:/해상 객체 이미지/Validation/"

total_num=0
sub_num=0
flag1=0
flag2=0

'''
with tf.io.TFRecordWriter("validation.tfrecord") as f:
    for a,b,c in zip(file_name_valid[:7],valid_origin_list[:7],valid_label_list[:7]):
        for d in a:
            if str.split(d,'.')[0] in os.listdir("./valid"):
                continue
            else:
                shutil.copy2(source+d, "./valid")
                unzip("./valid/"+d,"./valid/"+str.split(d,'.')[0])
                os.remove("./valid/"+d)
        print("Unzip & Remove zip file in one port\n")
        c=c*len(b)
        print("Start Extract data in one port\n")
        sub_num=0
        flag1=0
        for pt,pl in zip(b,c):
            if (flag1==1)|(flag2==1):
                break
            xml_path1=pl+'/'+os.listdir(pl)[0]+'/'
            for t1 in os.listdir(pt):
                if (flag1==1)|(flag2==1):
                    break
                for i in os.listdir(pt+'/'+t1):
                    if (flag1==1)|(flag2==1):
                        break
                    subpath1=t1+'/'+i+'/'    
                    for j in os.listdir(pt+'/'+subpath1):
                        if (flag1==1)|(flag2==1):
                            break
                        subpath2=subpath1+j+'/'
                        rpath=os.listdir(pt+'/'+subpath2)
                        random.shuffle(rpath)
                        jump_var=0
                        for k in rpath:
                            if (flag1==1)|(flag2==1):
                                break
                            if jump_var>0:
                                jump_var=jump_var-1
                                continue
                            data_path=subpath2+k
                            full_path=pt+'/'+data_path
                            fp="/"+str.join('/',str.split(full_path,'/')[4:])
                            dic=fetch_data2(full_path,xml_path1,fp)
                            if dic:
                                jump_var=2
                                result=serialize_example(dic)
                                f.write(result)
                                total_num=total_num+1
                                sub_num=sub_num+1
                                if (sub_num>=200)|(total_num>=1000):
                                    flag1=1
                                    if total_num>=1000:
                                        flag2=1
                                    break
                                
                                                
            shutil.rmtree(pt)
        shutil.rmtree(pl)
        print("End Extract data in one port\n")
'''


valid_origin_list=[]
valid_label_list=[]
for t in file_name_valid:
    subo_list=[]
    subl_list=[]
    for k in t:
        if re.search("원천",k):
            subo_list.append("./test/"+str.split(k,".zip")[0])
        if re.search("라벨",k):
            subl_list.append("./test/"+str.split(k,".zip")[0])
    valid_origin_list.append(subo_list)
    valid_label_list.append(subl_list)

file_name_valid[:7]
valid_origin_list[:7]
valid_label_list[:7]



total_num=0
sub_num=0
flag1=0
flag2=0

'''
with tf.io.TFRecordWriter("test.tfrecord") as f:
    for a,b,c in zip(file_name_valid[7:],valid_origin_list[7:],valid_label_list[7:]):
        for d in a:
            if str.split(d,'.')[0] in os.listdir("./test"):
                continue
            else:
                shutil.copy2(source+d, "./test")
                unzip("./test/"+d,"./test/"+str.split(d,'.')[0])
                os.remove("./test/"+d)
        print("Unzip & Remove zip file in one port\n")
        c=c*len(b)
        print("Start Extract data in one port\n")
        sub_num=0
        flag1=0
        for pt,pl in zip(b,c):
            if (flag1==1)|(flag2==1):
                break
            xml_path1=pl+'/'+os.listdir(pl)[0]+'/'
            for t1 in os.listdir(pt):
                if (flag1==1)|(flag2==1):
                        break
                for i in os.listdir(pt+'/'+t1):
                    if (flag1==1)|(flag2==1):
                        break
                    subpath1=t1+'/'+i+'/'    
                    for j in os.listdir(pt+'/'+subpath1):
                        if (flag1==1)|(flag2==1):
                            break
                        subpath2=subpath1+j+'/'
                        rpath=os.listdir(pt+'/'+subpath2)
                        random.shuffle(rpath)
                        jump_var=0
                        for k in rpath:
                            if (flag1==1)|(flag2==1):
                                break
                            if jump_var>0:
                                jump_var=jump_var-1
                                continue
                            data_path=subpath2+k
                            full_path=pt+'/'+data_path
                            fp="/"+str.join('/',str.split(full_path,'/')[4:])
                            dic=fetch_data2(full_path,xml_path1,fp)
                            if dic:
                                jump_var=2
                                result=serialize_example(dic)
                                f.write(result)
                                total_num=total_num+1
                                sub_num=sub_num+1
                                if (sub_num>=200)|(total_num>=1000):                                    
                                    flag1=1
                                    if total_num>=1000:
                                        flag2=1
                                    break
                                                
            shutil.rmtree(pt)
        shutil.rmtree(pl)
        print("End Extract data in one port\n")
'''


# check TFRecord File

def load_fetched_dataset(save_dir):
    train = tf.data.TFRecordDataset(f"{save_dir}/train.tfrecord".encode("utf-8")).map(
        deserialize_example
    )
    validation = tf.data.TFRecordDataset(
        f"{save_dir}/validation.tfrecord".encode("utf-8")
    ).map(deserialize_example)
    test = tf.data.TFRecordDataset(f"{save_dir}/test.tfrecord".encode("utf-8")).map(
        deserialize_example
    )

    return train, validation, test


path="./aihub/"
train,valid,test=load_fetched_dataset(path)
train2=train.batch(2)

for i in train2:
    print(decode_filename(i['filename']))




