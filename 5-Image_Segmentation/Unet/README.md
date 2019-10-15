
## U-Net: Convolutional Networks for Biomedical Image Segmentation.
--------------------
 [this paper](https://arxiv.org/abs/1505.04597) presents a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.

## Train membrane
--------------------
[membrane](https://github.com/YunYang1994/membrane) contains 90 images for training and 30 for testing.  The corresponding binary labels are provided in an in-out fashion, i.e. white for the pixels of segmented objects and black for the rest of pixels (which correspond mostly to membranes)

| input | ground truth | prediction |
|---|---|:---:|
|![image](./results/origin_0.png)|![image](./results/gt_0.png)|![image](./results/pred_0.png)|
|![image](./results/origin_1.png)|![image](./results/gt_1.png)|![image](./results/pred_1.png)|

Finally, you can train it and then evaluate your model

```bashrc
$ git clone https://github.com/YunYang1994/membrane.git
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
@Tensorflow2.0-Examples{,
  author       = YunYang1994,
  email        = www.dreameryangyun@sjtu.edu.cn,
  title        = "U-Net: Convolutional Networks for Biomedical Image Segmentation",
  url          = https://github.com/YunYang1994/TensorFlow2.0-Examples,
  year         = 2019,
}
```
