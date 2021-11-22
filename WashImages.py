import cv2
import os
from utils.retinaface import RetinaFace

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'

thresh = 0.8

gpuid = 0
detector = RetinaFace('./model/mnet.25/mnet.25', 0, gpuid)

img_dir = './DATA/Images/'

del_list = []

step = 0
for img_name in os.listdir(img_dir):

    img_path = os.path.join(img_dir, img_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(640,640))

    faces, _ = detector.detect(img, thresh)

    if len(faces) == 0:
        del_list.append(img_name)

    step += 1
    if step%500 == 0:
        print('On step: {}/{}'.format(step, len(os.listdir(img_dir))))

print('Find {} images without faces.'.format(len(del_list)))
action = input('Delete them? (Y/N)')

if action == 'Y':
    for img_name in del_list:
        img_path = os.path.join(img_dir, img_name)
        os.remove(img_path)
    print('Finished Deletion. Bye!')
else:
    print('I did not do it. Bye!')