# VGG16
![image](./docs/vgg16.png)

This is a Tensorflow implemention of [VGG16: Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf). Original Caffe implementation can be found in [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)

## Usage
To use the VGG networks, the [weight files](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) for VGG16 has to be downloaded and put it in the root path .The `images` is a tensor with shape `[None, 224, 224, 3]`.

```
$ python vgg16.py
n02123045 tabby, tabby cat
```

![image](./docs/cat.jpg)

## Keypoints
- **No BatchNorm layers**
- **The first one to use small kernel-sized filters**
- **The network architecture weights themselves are quite large (553MB)**
- **It is painfully slow to train**
