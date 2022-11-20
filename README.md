# yolov7-pose
Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806). 

## Dataset preparison

[[Keypoints Labels of nimbro in COCO format]](https://www.ais.uni-bonn.de/~hfarazi/RC2021/hrp.zip)
``` shell
python coco2yolo.py -i '/mnt/d/Reza/Dokumen/datasets/robot_pose/robot_keypoints_train_fixed.json' -s '/mnt/d/Reza/Dokumen/datasets/robot_pose/images/' -t "yolo_robot_pose" -n "train"
```

## Training

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

``` shell
python train.py --data data/robot_kpts.yaml --cfg cfg/yolov7-w6-robot-pose.yaml --weights weights/yolov7-w6-pose.pt --batch-size 2 --kpt-label --device 0 --name yolov7-w6-pose --hyp data/hyp.pose.yaml
```

## Deploy
TensorRT:[https://github.com/nanmi/yolov7-pose](https://github.com/nanmi/yolov7-pose)

## Testing

[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

``` shell
python test.py --data data/coco_kpts.yaml --img 960 --conf 0.001 --iou 0.65 --weights yolov7-w6-pose.pt --kpt-label
```

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
