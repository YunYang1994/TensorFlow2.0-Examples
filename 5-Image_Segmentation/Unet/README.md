
## [U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://yunyang1994.gitee.io/2018/11/12/Unet/)
--------------------
 [this paper](https://arxiv.org/abs/1505.04597) presents a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.

## Train membrane
--------------------
Membrane contains 90 images for training and 30 for testing.  The corresponding binary labels are provided in an in-out fashion, i.e. white for the pixels of segmented objects and black for the rest of pixels (which correspond mostly to membranes)

<p align="center">
    <img width="37%" src="https://user-images.githubusercontent.com/30433053/67922238-2ba7a380-fbe5-11e9-96a0-55c6827bd024.png" style="max-width:37%;">
    </a>
</p>


you can download Membrane dataset on the [BaiduCloud Drive](https://pan.baidu.com/s/1O9U48zDjulhLYONksX569w) and put it in `./`, then you can train it and then evaluate your model

```bashrc
$ python train.py

Epoch 1/5
Found 90 images belonging to 1 classes.
Found 90 images belonging to 1 classes.
5000/5000 [==============================] - 1443s 289ms/step - loss: 0.1926 - accuracy: 0.9456
Epoch 2/5
5000/5000 [==============================] - 1438s 288ms/step - loss: 0.1026 - accuracy: 0.9874
...
=> accuracy: 0.7934, saving ./results/pred_0.png
=> accuracy: 0.8132, saving ./results/pred_1.png
...
```

## Citation
--------------------
```
@Github_Project{TensorFlow2.0-Examples,
  author       = YunYang1994,
  email        = www.dreameryangyun@sjtu.edu.cn,
  title        = "U-Net: Convolutional Networks for Biomedical Image Segmentation",
  url          = https://github.com/YunYang1994/TensorFlow2.0-Examples,
  year         = 2019,
}
```
