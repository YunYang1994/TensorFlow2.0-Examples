# TensorFlow2.x-YOLOv3
A minimal tensorflow implementation of YOLOv3, with support for training, inference and evaluation.

## Installation

Install requirements and download pretrained weights

```
$ pip3 install -r ./docs/requirements.txt
$ wget https://pjreddie.com/media/files/yolov3.weights
```

## Quick start

```
$ python image_demo.py
$ python video_demo.py # if use camera, set video_path = 0
```
![image](./docs/kite_result.jpg)

## Train yymnist

Download [yymnist](https://github.com/YunYang1994/yymnist) dataset and make data.

```
$ git clone https://github.com/YunYang1994/yymnist.git
$ python yymnist/make_data.py --images_num 1000 --images_path ./dataset/train --labels_txt ./dataset/yymnist_train.txt
$ python yymnist/make_data.py --images_num 200  --images_path ./dataset/test  --labels_txt ./dataset/yymnist_test.txt
```


