## [MTCNN: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://github.com/YunYang1994/ai-notebooks/blob/master/MTCNN.md)
--------------------
This repository is the implementation of MTCNN in TF2.  It is written from scratch, using as a reference the implementation of MTCNN from David Sandberg Project: [facenet](https://github.com/davidsandberg/facenet).

## Inference
Just download the repository and then do this

```bashrc
$ python main.py
```
<p align="center">
    <img width="100%" src="https://user-images.githubusercontent.com/30433053/67630296-52519b80-f8c0-11e9-8185-75f2e0d46ee8.png" style="max-width:100%;">
    </a>
</p>

The detection results contains three part:

- **Face classification**: the probability produced by the network that indi- cates a sample being a face
- **Bounding box regression**: the bounding boxes’ left top, height, and width.
- **Facial landmark localization**: There are five facial landmarks, including left eye, right eye, nose, left mouth corner, and right mouth corner

## Citation

```bashrc
@Github_Project{TensorFlow2.0-Examples,
  author       = YunYang1994,
  email        = www.dreameryangyun@sjtu.edu.cn,
  title        = "MTCNN: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks",
  url          = https://github.com/YunYang1994/TensorFlow2.0-Examples,
  year         = 2019,
}
```




