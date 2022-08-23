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


#unzip("./[라벨]동해_묵호항_1구역_BOX.zip","./")
#unzip("./제주항_맑음_20201227_0848_0004.zip","../")
#unzip("./[라벨]서해_군산항_1구역_BOX.zip","./")

# 이 부분 수정해서 되게끔 만들기.

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


def decode_str(strtensor):
    raw_string = str(strtensor.numpy(),'utf-8')
    return raw_string


def parse_func(example):
    image = tf.io.decode_raw(example["image"], tf.float32)
    image_shape = tf.io.decode_raw(example["image_shape"], tf.int32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    bbox_shape = tf.io.decode_raw(example["bbox_shape"], tf.int32)
    image=tf.reshape(image,image_shape)
    bbox=tf.reshape(bbox,bbox_shape)
    filename=tf.py_function(decode_str,[example["filename"]],[tf.string])

    return {'image':image,'bbox':bbox,'filename':filename}

def deserialize_example(serialized_string):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_shape": tf.io.FixedLenFeature([], tf.string),
        "bbox": tf.io.FixedLenFeature([],tf.string),
        "bbox_shape": tf.io.FixedLenFeature([],tf.string),
        "filename": tf.io.FixedLenFeature([],tf.string)
    }
    example = tf.io.parse_example(serialized_string, image_feature_description)
    
    dataset=tf.data.Dataset.from_tensor_slices(example).map(parse_func)
        
    return dataset


def fetch_data(path):
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


pt="./[원천]남해_여수항_2구역_BOX/남해_여수항_2구역_BOX/"
#data=[]

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
                   

# 모든 zip 파일에 대해서 새론 폴더로 옮기고(짝 맞춰서), unzip하고 다 처리되면 파일 삭제 
# 우선 training에 있는 파일들만 사용해서 train.tfrecord로 모으기
# 여기서, 한글 파일명은 일단은 utf-8로 바뀌어서 나오는데, eager tensor에서는 무슨 이유인지는 몰라도
# 계속 바뀌지 않는 오류가 발생함. 따라서, 우선 나오도록하고, 추후에 str(filename,'utf-8) 로 바꾸어서 
# 출력하는 것이 최선이라고 판단함.

'''
dataset=tf.data.TFRecordDataset(["test.tfrecord"]) 
tmp=iter(dataset)
dataset
n_data=0
for i in dataset.batch(100000,drop_remainder=True).take(1):
    n_data=n_data+1

dataset=tf.data.TFRecordDataset(["test.tfrecord"]).batch(1)

for i in dataset:
    example = deserialize_example(i)
    break

for v in example:
    print(v)
'''

c1=os.listdir("E:/해상 객체 이미지/Training")


path="E:/해상 객체 이미지/Training"
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

train_fname=filename_extract("E:/해상 객체 이미지/Training")
        
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


file_name=find_file(path,train_fname)


os.getcwd()
os.chdir("C:/Users/UOS/Desktop/새론")

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



file_name
origin_list
label_list

# example
os.getcwd()

def fetch_data2(image_path,xml_path1,xml_path2):
    obj_num=0
    image_path= image_path
    xml_path_ = xml_path1+str.split(xml_path2,'.')[0]+".xml"
    
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


str.split(file_name[0][0],'.')[0]

str.split(file_name[0][0],'.')[0] in os.listdir("./train")

source = "E:/해상 객체 이미지/Training/"
with tf.io.TFRecordWriter("training.tfrecord") as f:
    for a,b,c in zip(file_name,origin_list,label_list):
        for d in a:
            if str.split(d,'.')[0] in os.listdir("./train"):
                continue
            else:
                shutil.copy2(source+d, "./train")
                unzip("./train/"+d,"./train/"+str.split(d,'.')[0])
                os.remove("./train/"+d)
        print("Unzip & Remove zip file in one place\n")
        c=c*len(b)
        print("Start Extract data in one place\n")
        for pt,pl in zip(b,c):
            xml_path1=pl+'/'+os.listdir(pl)[0]+'/'
            for t1 in os.listdir(pt):
                for i in os.listdir(pt+'/'+t1):
                    subpath1=t1+'/'+i+'/'    
                    for j in os.listdir(pt+'/'+subpath1):
                        subpath2=subpath1+j+'/'
                        for k in os.listdir(pt+'/'+subpath2):
                            data_path=subpath2+k
                            full_path=pt+'/'+data_path
                            fp="/"+str.join('/',str.split(full_path,'/')[4:])
                            dic=fetch_data2(full_path,xml_path1,fp)
                            if dic:
                                result=serialize_example(dic)
                                f.write(result)                
            shutil.rmtree(pt)
        shutil.rmtree(pl)
        print("End Extract data in one place\n")

            
























