import os
import xml.etree.ElementTree as Et
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zipfile
import json

os.getcwd()
os.chdir("C:/Users/UOS/Desktop/새론")

def unzip(source_file, dest_path):
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            member.filename = member.filename.encode("cp437").decode("euc-kr")
            if str.split(member.filename,'.')[-1]=="xml":
                zf.extract(member,dest_path)


#unzip("./[라벨]남해_여수항_2구역_BOX.zip","./")
#unzip("./제주항_맑음_20201227_0848_0004.zip","../")

"제주항_맑음_20201227_0848_0004.zip".encode('utf-8')


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
    print(raw_string)
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
    image_shape=np.array([500,500,3])
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
        img2=tf.image.resize(img2,(500,500))
        bbox_shape=np.array([obj_num,4])

        return {"filename":img_name,"image":img2.numpy(),"image_shape":image_shape,"bbox":bndbox_list.numpy(),"bbox_shape":bbox_shape}

    else:
        return None


pt="./[원천]남해_여수항_2구역_BOX/남해_여수항_2구역_BOX/"
write_path="./train/남해_여수항_2구역_BOX.json"
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
                break    



dataset=tf.data.TFRecordDataset(["test.tfrecord"]).batch(3)

for i in dataset:
    example = deserialize_example(i)

for v in example:
    print(v)




