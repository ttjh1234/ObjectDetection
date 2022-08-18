import os
from quopri import decodestring
import xml.etree.ElementTree as Et
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zipfile

os.getcwd()

def unzip(source_file, dest_path):
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            member.filename = member.filename.encode("cp437").decode("euc-kr")
            if str.split(member.filename,'.')[-1]=="xml":
                zf.extract(member,dest_path)


#unzip("./[라벨]남해_여수항_2구역_BOX.zip","./")
#unzip("./제주항_맑음_20201227_0848_0004.zip","../")


def fetch_data(path):
    obj_num=0
    xml_path_  =  path
    xml =  open(xml_path_, mode = 'r', encoding="utf-8")
    xml_tree = Et.parse(xml) 
    xml_root = xml_tree.getroot() 
    img_name=xml_root.find("filename").text
    objects = xml_root.findall("object")

    bndbox_list=[]

    for i in objects:
        if i.find("category_id").text==2:
            obj_num=obj_num+1
            bndbox=i.find("bndbox") # object 한 객체내에 bndbox 접근
            xmin=bndbox.find('xmin').text # x최소 좌표 
            xmax=bndbox.find('xmax').text# x최대 좌표
            ymin=bndbox.find('ymin').text # y최소 좌표 
            ymax=bndbox.find('ymax').text# y최대 좌표
            bndbox_list.append([ymin,xmin,ymax,xmax])

    if obj_num>=3:
        image_path='./[원천]남해_여수항_2구역_BOX/'+img_name
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2=tf.cast(img,dtype=tf.float64)

        return {"filename":img_name,"image":img2,"bbox":bndbox_list}

    else:
        return None



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


















