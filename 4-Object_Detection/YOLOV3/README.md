## [TensorFlow2.x-YOLOv3](https://yunyang1994.github.io/posts/YOLOv3/#more)
--------------------
A minimal tensorflow implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
--------------------
Install requirements and download pretrained weights

```
$ pip3 install -r ./docs/requirements.txt
$ wget https://pjreddie.com/media/files/yolov3.weights
```

## Quick start
--------------------
In this part, we will use pretrained weights to make predictions on both image and video.

```
$ python image_demo.py
```

<p align="center">
    <img width="100%" src="https://user-images.githubusercontent.com/30433053/68088581-9255e700-fe9b-11e9-8672-2672ab398abe.jpg" style="max-width:100%;">
    </a>
</p>

## Train yymnist
--------------------

<p align="center">
    <img width="70%" src="https://user-images.githubusercontent.com/30433053/68088705-90d8ee80-fe9c-11e9-8e61-589fdc45fe60.png" style="max-width:70%;">
    </a>
</p>



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
$ tensorboard --logdir ./data/log
$ python test.py
$ cd ../mAP
$ python main.py        # Detection images are expected to save in `YOLOV3/data/detection`
```
Track training progress in Tensorboard and go to http://localhost:6006/

```
$ tensorboard --logdir ./data/log
```
<p align="center">
    <img width="100%" src="https://user-images.githubusercontent.com/30433053/68088727-db5a6b00-fe9c-11e9-91d6-555b1089b450.png" style="max-width:100%;">
    </a>
</p>

## Citation
--------------------
```
@Github_Project{TensorFlow2.0-Examples,
  author       = YunYang1994,
  email        = www.dreameryangyun@sjtu.edu.cn,
  title        = "YOLOv3: An Incremental Improvement",
  url          = https://github.com/YunYang1994/TensorFlow2.0-Examples,
  year         = 2019,
}
```

