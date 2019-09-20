![image](https://user-images.githubusercontent.com/30433053/65319751-ee59fa00-dbd2-11e9-9065-10629dfd6d2f.png)

# Train VOC dataset
Download VOC PASCAL trainval and test data
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
|       └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
             └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
                     └──VOC2012 (from VOCtrainval_11-May-2012.tar)
```

Then you need to parser VOC dataset and train it

```bashrc
$ python parser_voc.py --voc_path /home/yang/dataset/VOC
$ python train.py
```