## [RPN: RegionProposal Network](https://github.com/YunYang1994/Easy-Deep-Learning/blob/master/RPN.md)
--------------------

This repository is implemented for the paper ["Expecting the Unexpected: Training Detectors for Unusual Pedestrians with Adversarial Imposters(CVPR2017)"](https://arxiv.org/pdf/1703.06283), which makes some improvements on the basis of region proposal network in FasterRCNN. 

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/66986053-a904eb80-f0f0-11e9-93fa-c56fb580f6ae.png" style="max-width:80%;">
    </a>
</p>

## Synthetic datasets
--------------------

For training RPN, I use synthetic datasets which contains a set of 8239 images with one class (`pedestrian`). The data is available on the [BaiduCloud](https://pan.baidu.com/s/1QZAIakMVS0sJV0sjgv7v2w&shfl=sharepset). We isotropically scale each image to a resolution of 960×720, zero-padding as necessary.

<p align="center">
    <img width="70%" src="https://user-images.githubusercontent.com/30433053/67255940-306aaa00-f4b7-11e9-997b-f3cbeb249191.png" style="max-width:70%;">
    </a>
</p>

## Anchors Demo
--------------------
I think the most exciting things in the region proposal network is Anchors. I use 9 anchors at each sliding position. During training, a candidate bounding box will be treated as a positive if its intersection-over-union overlap with a ground-truth box exceeds 50%, and will be a negative for overlaps less than 10%. 

```bashrc
$ python demo.py
```

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/67204319-db3f8180-f43f-11e9-99fe-bb73b0123fc6.png" style="max-width:60%;">
    </a>
</p>

> In the above picture, the blue anchor boxes are positive samples. the red and black dot represented for the center of positive, negative samples respectively.

## Network Training
--------------------

First of all, you need to download [synthetic datasets](https://pan.baidu.com/s/1QZAIakMVS0sJV0sjgv7v2w&shfl=sharepset) and put it in `./`, then

```bashrc
$ python train.py
```
the model will automatically save weights `RPN.h5` in each epoch. Finally you can test `RPN.h5` with test.py


## Citation
--------------------
```bashrc
@Github_Project{Tensorflow2.0-Examples,
  author       = YunYang1994,
  email        = www.dreameryangyun@sjtu.edu.cn,
  title        = "RPN: a Region Proposal Network",
  url          = https://github.com/YunYang1994/TensorFlow2.0-Examples,
  year         = 2019,
}
```



