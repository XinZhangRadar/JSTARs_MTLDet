
from HTL.canny import edge_gen
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
import pdb
from tqdm import tqdm
def generate_edge(image_info):
    edge_save_name = '/share/home/zhangxin/mmdetection/data/coco_hrsid/edge_ori/' + image_info['filename'] + '_edge.jpg'
    if os.path.exists(edge_save_name): 
        return
    bboxes = image_info['gt_bboxes']

    img = image_info['img']
    edge,edge_mask,mask = edge_gen(img,bboxes)
    cv2.imwrite(edge_save_name,edge_mask)



def _load_pascal_annotation(data_path,imagename):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    image_path = os.path.join(data_path, 'JPEGImages', imagename + '.jpg')
    img = cv2.imread(image_path)
    anno_path = os.path.join(data_path, 'Annotations', imagename + '.xml')
    #pdb.set_trace()
    tree = ET.parse(anno_path)
    objs = tree.findall('object')

    num_objs = len(objs)
    
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)

    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text) 
        boxes[ix, :] = [x1, y1, x2, y2]
    return {'gt_bboxes': boxes,'img':img,'filename':imagename}




data_path = "/share/home/zhangxin/data/HRSID/"
image_names = os.listdir(data_path+'JPEGImages/')
for image_name in tqdm(image_names):
    imagename = image_name.split('.')[0]
    image_info = _load_pascal_annotation(data_path,imagename)
    generate_edge(image_info)

