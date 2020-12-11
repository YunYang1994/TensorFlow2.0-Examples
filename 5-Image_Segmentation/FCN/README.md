## [FCN: Fully Convolutional Networks for Semantic Segmentation](https://yunyang1994.gitee.io/2019/07/12/FCN/)
--------------------
 [This paper](https://arxiv.org/abs/1411.4038) shows that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. This repo adapts vgg16 networks into fully convolutional networks and transfer their learned representations by fine-tuning to the segmentation task on PASCAL VOC2012 dataset. 
 
<p align="center">
    <img width="50%" src="https://user-images.githubusercontent.com/30433053/67369222-33df5d80-f5ab-11e9-95d4-3d7813cfa0a8.png" style="max-width:50%;">
    </a>
</p>

## Train PASCAL VOC2012
--------------------
Download VOC PASCAL trainval and test data. 

```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory and rename them, which should have the following basic structure.

```bashrc
VOC           # path:  /home/yang/dataset/VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
```
Then you need parser_voc.py to parser PASCAL VOC dataset and train it. Finally you can evaluate model by `python test.py`. The prediction results are expected to save in `./data/prediction`. Here is my trained weight [FCN8s.h5](https://pan.baidu.com/s/1EUQNBHZmS6w40y4A3n_TEg)

```bashrc
$ python parser_voc.py --voc_path /home/yang/dataset/VOC
$ python train.py
Epoch 1/30
6000/6000 [==============================] - 3571s 595ms/step - loss: 1.2374 - accuracy: 0.7233
Epoch 2/30
6000/6000 [==============================] - 3552s 592ms/step - loss: 1.1319 - accuracy: 0.7497
...
Epoch 30/30
6000/6000 [==============================] - 3552s 592ms/step - loss: 0.0811 - accuracy: 0.9797

$ python test.py
```

|![image](https://user-images.githubusercontent.com/30433053/66732790-d4d56680-ee8f-11e9-9120-07b0e8aa53d4.jpg)|![image](https://user-images.githubusercontent.com/30433053/66732791-d69f2a00-ee8f-11e9-9c5d-16cc84bc7e9e.jpg)|![image](https://user-images.githubusercontent.com/30433053/66732795-da32b100-ee8f-11e9-9d85-f0ddba7a3ab1.jpg)|
|---|---|:---:|
|![image](https://user-images.githubusercontent.com/30433053/66732799-dd2da180-ee8f-11e9-9025-3a3e0e94a20b.jpg)|![image](https://user-images.githubusercontent.com/30433053/66733895-aa85a800-ee93-11e9-8eae-405235aa8564.jpg)|![image](https://user-images.githubusercontent.com/30433053/66733897-ace80200-ee93-11e9-84e4-21f7d94d06eb.jpg)|

## Citation
--------------------
```
@Github_Project{TensorFlow2.0-Examples,
  author       = YunYang1994,
  email        = www.dreameryangyun@sjtu.edu.cn,
  title        = "FCN: Fully Convolutional Networks for Semantic Segmentation.",
  url          = https://github.com/YunYang1994/TensorFlow2.0-Examples,
  year         = 2019,
}
```

