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

'''
# 이 부분 수정해서 되게끔 만들기.

def serialize_example(dic):
    image = dic["image"].tobytes()
    bbox = dic["bbox"].tobytes()
    filename = dic["filename"].tobytes()

    feature_dict = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        "bbox": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
        "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename]))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example.SerializeToString()

def deserialize_example(serialized_string):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "bbox": tf.io.FixedLenFeature([], tf.string),
        "filename": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(serialized_string, image_feature_description)

    image = tf.io.decode_raw(example["image"], tf.float32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    filename = tf.io.decode_raw(example["filename"], tf.int32)

    return image, bbox, filename

'''

def fetch_data(path):
    obj_num=0
    image_path= "./[원천]남해_여수항_2구역_BOX/남해_여수항_2구역_BOX/"+path
    xml_path_ = "./[라벨]남해_여수항_2구역_BOX/남해_여수항_2구역_BOX/"+str.split(path,'.')[0]+".xml"
    
    xml =  open(xml_path_, mode = 'r', encoding="utf-8")
    xml_tree = Et.parse(xml) 
    xml_root = xml_tree.getroot()
    img_name=np.array(xml_root.find("filename").text)
    objects = xml_root.findall("object")
    size = xml_root.find("size") 
    width = int(size.find("width").text)  
    height= int(size.find("height").text)
    image_shape=np.array((height,width,3))

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

        return {"filename":img_name,"image":img2.numpy(),"image_shape":image_shape,"bbox":bndbox_list.numpy()}

    else:
        return None


pt="./[원천]남해_여수항_2구역_BOX/남해_여수항_2구역_BOX/"
write_path="./train/남해_여수항_2구역_BOX.json"
data=[]

data[0]['filename'].tobytes()
data[0]['image_shape'].tobytes()
data[0]['image'].tobytes()
data[0]['bbox']

for i in os.listdir(pt):
    subpath1=i+'/'    
    for j in os.listdir(pt+subpath1):
        subpath2=subpath1+j+'/'
        for k in os.listdir(pt+subpath2):
            data_path=subpath2+k
            dic=fetch_data(data_path)
            if dic:
                data.append(dic)
                '''
                result=serialize_example(dic)
                with tf.io.TFRecordWriter("test.tfrecord") as f:
                    f.write(result)
                '''
            break    

data

dataset=tf.data.TFRecordDataset(["test.tfrecord"]).batch(10)
for i in dataset:
    print(deserialize_example(i))

data



'''
# 이전 test 코드

xml_path_  =  "./제주항_맑음_20201227_0848_0004.xml"
xml =  open(xml_path_, mode = 'r', encoding="utf-8")
xml_tree = Et.parse(xml) 
xml_root = xml_tree.getroot() 
img_name=xml_root.find("filename").text
size = xml_root.find("size") 
width = size.find("width")  
height=size.find("height") 
objects = xml_root.findall("object") # object들에 접근 , 다수가 존재할 수 있으므로 findall 사용


image_path="./제주항_맑음_20201227_0848_0004.jpg"
img_array = np.fromfile(image_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2=tf.cast(img,dtype=tf.float64)

def vision_valid(image,gt_box):
    img_rgb_copy = image.copy()
    green_rgb = (255, 0, 0)
    for rect in gt_box:
        left = rect[1]
        top = rect[0]
        # rect[2], rect[3]은 너비와 높이이므로 우하단 좌표를 구하기 위해 좌상단 좌표에 각각을 더함.
        right = rect[3]
        bottom = rect[2]        

        img_rgb_copy = cv2.rectangle(img_rgb_copy, (int(left), int(top)), (int(right), int(bottom)), color=green_rgb, thickness=5)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb_copy)
    plt.show()


bndbox_list=[]

for i in objects:
    bndbox=i.find("bndbox") # object 한 객체내에 bndbox 접근
    xmin=bndbox.find('xmin').text # x최소 좌표 
    xmax=bndbox.find('xmax').text# x최대 좌표
    ymin=bndbox.find('ymin').text # y최소 좌표 
    ymax=bndbox.find('ymax').text# y최대 좌표
    bndbox_list.append([ymin,xmin,ymax,xmax])
    
vision_valid(img,bndbox_list)

len(objects)


'''







