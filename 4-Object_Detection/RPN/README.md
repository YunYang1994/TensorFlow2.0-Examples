## [RPN: RegionProposal Network](https://yunyang1994.gitee.io/2019/09/27/RPN/)
--------------------

This repository is implemented for paper ["Expecting the Unexpected: Training Detectors for Unusual Pedestrians with Adversarial Imposters(CVPR2017)"](https://arxiv.org/pdf/1703.06283), which makes some improvements on the basis of region proposal network in [Faster-RCNN](http://arxiv.org/abs/1506.01497). 

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/66986053-a904eb80-f0f0-11e9-93fa-c56fb580f6ae.png" style="max-width:80%;">
    </a>
</p>

## Synthetic datasets
--------------------

For training RPN, I use synthetic datasets which contains a set of 8239 images with one class (`pedestrian`). The data is available on [BaiduCloud](https://pan.baidu.com/s/1QZAIakMVS0sJV0sjgv7v2w&shfl=sharepset). We isotropically scale each image to a resolution of 960×720, zero-padding as necessary.

<p align="center">
    <img width="70%" src="https://user-images.githubusercontent.com/30433053/67255940-306aaa00-f4b7-11e9-997b-f3cbeb249191.png" style="max-width:70%;">
    </a>
</p>

**First of all, you need to download** [synthetic datasets](https://pan.baidu.com/s/1QZAIakMVS0sJV0sjgv7v2w&shfl=sharepset) and put it in `./`

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

> In the above picture, the blue anchor boxes are positive samples. the red and black dots represent for the center position of positive, negative samples respectively.

9 anchors are generated by k-means algorithmn. This process is optional since I have integrated them in my code. But if you are not familiar with it, you can have a look it by

```bashrc
$ python kmeans.py
```

## Network Training
--------------------

```bashrc
$ python train.py
```
The model will automatically save weights `./RPN.h5` in each epoch. During neural network training， you can track the loss curve in Tensorboard and go to http://localhost:6006/ 

|score loss|boxes loss|total loss
|:---:|:---:|:---:
|![image](https://user-images.githubusercontent.com/30433053/67396155-acf5a980-f5d9-11e9-85cf-0dc1642aa552.png)|![image](https://user-images.githubusercontent.com/30433053/67396100-994a4300-f5d9-11e9-9a08-561ab5ee5caf.png)|![image](https://user-images.githubusercontent.com/30433053/67396192-baab2f00-f5d9-11e9-886a-d0f2994ba401.png)


```bashrc
=> epoch 1  step 1  total_loss: 0.402951  score_loss: 0.346327  boxes_loss: 0.056625
=> epoch 1  step 2  total_loss: 0.399650  score_loss: 0.344363  boxes_loss: 0.055287
......
=> epoch 10  step 4000  total_loss: 0.001989  score_loss: 0.000015  boxes_loss: 0.001973
```
Finally you can test `RPN.h5` with [test.py](https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/4-Object_Detection/RPN/test.py) for 200 images. The prediction results are expected to save in `./prediction`. Here is my trained weight https://pan.baidu.com/s/1o6HeQou9sCczLA_rzkTPqg

```
$ python test.py
```

|![image](https://user-images.githubusercontent.com/30433053/67265442-5789a180-f4e0-11e9-9fcd-6e72136c2913.png)|![image](https://user-images.githubusercontent.com/30433053/67265549-915aa800-f4e0-11e9-91e8-87ee05b7748c.png)|
|---|---
|![image](https://user-images.githubusercontent.com/30433053/67265487-6c663500-f4e0-11e9-8bf9-f9d59d22b0a8.png)|![image](https://user-images.githubusercontent.com/30433053/67265620-c36c0a00-f4e0-11e9-8689-9d3b6efaff47.png)

## Citation
--------------------
```bashrc
@Github_Project{TensorFlow2.0-Examples,
  author       = YunYang1994,
  email        = www.dreameryangyun@sjtu.edu.cn,
  title        = "RPN: a Region Proposal Network",
  url          = https://github.com/YunYang1994/TensorFlow2.0-Examples,
  year         = 2019,
}
```



