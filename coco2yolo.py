import sys, os
import json
import pydash
import shutil
import numpy as np
import argparse
from tqdm import tqdm

def __parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Path to anotation file")
    parser.add_argument("-s", "--source_images_path", help="Path to images directory")
    parser.add_argument("-t", "--target_path", help="Target folder to store anotation")
    parser.add_argument("-n", "--dataset_name", help="Dataset name")
    return parser.parse_args(args)

def xywh2xyxy_normalize(bbox, img_size):
    lx,ly,w,h = bbox
    x = lx+(w/2)
    y = ly+(h/2)
    
    width,height = img_size
    
    dw = 1.0 / width
    dh = 1.0 / height

    x *= dw
    w *= dw
    y *= dh
    h *= dh

    # return ["{:.6f}".format(x/width),"{:.6f}".format(y/height),"{:.6f}".format((x+w)/width),"{:.6f}".format((y+h)/height)]
    return ["{:.6f}".format(x),"{:.6f}".format(y),"{:.6f}".format(w),"{:.6f}".format(h)]

def keypoint_normalize(keypoints, img_size):
    kpts = []
    width,height = img_size
    for i in range(0, len(keypoints), 3):
        x,y,v = keypoints[i:i+3]
        kpts = kpts + ["{:.6f}".format(x/width),"{:.6f}".format(y/height), "{:.6f}".format(2)]
    return kpts

def transform_anotation(anotation, img_size):
    image_id = anotation['image_id']
    category_id = anotation['category_id']
    width,height = img_size
    
    bbox = xywh2xyxy_normalize(anotation['bbox'], [width,height])
    keypoints = keypoint_normalize(anotation['keypoints'], [width,height])
    
    return [category_id-1]+bbox+keypoints

if __name__ == '__main__':
    """
    Script to convert coco format anotation to yolo format.
    Input :
    -i = Path to anotation file
    -s = Path to images directory
    -t = Target folder to store anotation
    -n = Dataset name
    Example : python coco2yolo.py -i '/mnt/d/Reza/Dokumen/datasets/robot_pose/robot_keypoints_train_fixed.json' -s '/mnt/d/Reza/Dokumen/datasets/robot_pose/images/' -t "yolo_robot_pose" -n "train"
    """
    arguments = __parse_arguments(sys.argv[1:])
    input_file = arguments.input_file
    source_images_path = arguments.source_images_path
    target_path = arguments.target_path
    dataset_name = arguments.dataset_name

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    data = json.load( open(input_file) )

    target_images_path = os.path.join(target_path, "images", dataset_name)
    if os.path.exists(target_images_path):
        shutil.rmtree(target_images_path, ignore_errors=True)
    os.makedirs(target_images_path)

    target_anotation_path = os.path.join(target_path, "labels", dataset_name)
    if os.path.exists(target_anotation_path):
        shutil.rmtree(target_anotation_path, ignore_errors=True)
    os.makedirs(target_anotation_path)

    annotations = pydash.group_by(data['annotations'], 'image_id')

    with open(os.path.join(target_path, dataset_name+".txt"), 'w') as f1:
        for image in tqdm(data['images']):
            file_name = image['file_name']
            image_id = image['id']

            src = os.path.join(source_images_path, file_name)
            dst = os.path.join(target_images_path, file_name)
            shutil.copyfile(src, dst)

            f1.write(f"{os.path.join('./images', dataset_name, file_name)}\n")

            annotations_data = annotations[image_id]
        
            height = image['height']
            width = image['width']
            
            transform_results = []
            for anotation in annotations_data:
                result = transform_anotation(anotation, [width, height])
                transform_results.append(result)
            
            with open(os.path.join(target_anotation_path, format(image_id, '04d')+".txt"), 'w') as f2:
                for line in transform_results:
                    line = ' '.join(map(str, line))
                    f2.write(f"{line}\n")