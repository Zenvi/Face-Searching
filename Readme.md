# Search Images with Faces

## 1. Introduction

+ This repository is just a modified version of https://github.com/deepinsight/insightface
+ This repository achieves the function of searching images in a local file system using a face image

## 2. Environments

+ Python 3.6
+ MxNet with GPU support (Mine is CUDA10.2 + CuDnn7.6.5)
+ Opencv-Python
+ Numpy
+ Scikit-Image
+ Scikit-Learn
+ Go to https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA and download the insightface-R50 model, and put them in the `./model/insightface` folder and rename them to `InsightFace-0000.params`  and `InsightFace-symbol.json`

## 3. How to Replicate My Step

+ Simply run `face_searching.py`

+ If everything goes well, you will first observe the face image you used for searing, for example:

  ![2](.\mdPics\2.png)

+ And next you'll see the searching result(s), for example:

  ![Search results](.\mdPics\Search results.png)

## 4. How to Search in Your Own Dataset

+ Delete `./DATA/Images.json`
+ Put all your pictures in folder: `./DATA/Images` (Note: no subfolders shall exist in this folder, unless you modified the codes according to your own file structure)
+ Run `./utils/encode_dir.py`, a new `./DATA/Images.json` will be generated.
+ Modify the face image (used for searching) path in `face_searching.py` according to your own situation, and run `face_searching.py`

