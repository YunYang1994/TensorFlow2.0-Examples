# TensorFlow2.x-YOLOv3
A minimal tensorflow implementation of YOLOv3, with support for training, inference and evaluation.

## Installation

Install requirements and download pretrained weights

```
$ pip3 install -r ./docs/requirements.txt
$ wget https://pjreddie.com/media/files/yolov3.weights
```

## Quick start

In this part, we will use pretrained weights to make predictions on both image and video.

```
$ python image_demo.py
$ python video_demo.py # if use camera, set video_path = 0
```
![image](./docs/kite_result.jpg)

## Train yymnist

Download [yymnist](https://github.com/YunYang1994/yymnist) dataset and make data.

```
$ git clone https://github.com/YunYang1994/yymnist.git
$ python yymnist/make_data.py --images_num 1000 --images_path ./data/dataset/train --labels_txt ./data/dataset/yymnist_train.txt
$ python yymnist/make_data.py --images_num 200  --images_path ./data/dataset/test  --labels_txt ./data/dataset/yymnist_test.txt
```
Open `./core/config.py` and do some configurations
```
__C.YOLO.CLASSES                = "./data/classes/yymnist.names"
```

Finally, you can train it and then evaluate your model

```
$ python train.py
$ python test.py
$ cd ../mAP
$ python main.py        # Detection images are expected to save in `YOLOV3/data/detection`
```

| train |test|
|---|---
|![image](./docs/01554.jpg)|![image](./docs/01567.jpg)|

