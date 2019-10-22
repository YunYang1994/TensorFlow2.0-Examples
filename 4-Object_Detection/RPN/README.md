## RPN: RegionProposal Network

This repository is implemented for the paper ["Expecting the Unexpected: Training Detectors for Unusual Pedestrians with Adversarial Imposters(CVPR2017)"](https://arxiv.org/pdf/1703.06283), which makes some improvements on the basis of region proposal network in FasterRCNN. 

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/66986053-a904eb80-f0f0-11e9-93fa-c56fb580f6ae.png" style="max-width:80%;">
    </a>
</p>

## Synthetic datasets

For training RPN, I use synthetic datasets which contains a set of 8239 images with one class (`pedestrian`). The data is available on the [BaiduCloud](https://pan.baidu.com/s/1QZAIakMVS0sJV0sjgv7v2w&shfl=sharepset). As for image preprocessing, these images have been resized into the shape of `width: 720, height: 960`. 


