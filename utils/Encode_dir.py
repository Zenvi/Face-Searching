import cv2
import os
import json
import numpy as np
from utils.retinaface import RetinaFace
from utils.insightface import InsightFace

def images_2_json(image_dir, json_path):

    if os.path.exists(json_path):
        print('Target json already exists, skipping the conversion process...')
        return

    thresh = 0.8
    gpuid = 0

    print('Loading Models...')
    detector = RetinaFace('../model/mnet.25/mnet.25', 0, gpuid)
    Recognizer = InsightFace('../model/insightface/insightface', 0, gpuid)
    print('Done!')

    json_dict = {}

    print('Converting images...')
    step = 0
    for image_name in os.listdir(image_dir):

        image_path = os.path.join(image_dir, image_name)

        image = cv2.imread(image_path)

        faces, landmarks = detector.detect(image, thresh)

        vector_list = []
        if len(faces) == 0:
            json_dict[image_name] = vector_list
            continue

        for i in range(faces.shape[0]):

            face = faces[i].astype(np.int)
            landmark = landmarks[i].astype(np.int)

            aligned = Recognizer.prepare_insight_input(face, landmark, image)
            vector_list.append(Recognizer.face_vertorizing(aligned).tolist())

        json_dict[image_name] = vector_list

        step += 1

        if step%100 == 0:
            print( 'On {}/{}'.format(step, len(os.listdir(image_dir))) )

    print('Done image conversion, writing them to {}...'.format(json_path))
    with open(json_path, 'w') as f:
        f.write( json.dumps(json_dict, ensure_ascii=False, indent=4) )
    print('Written to {}'.format(json_path))

if __name__ == '__main__':
    image_dir ='F:/Leisure_Projects/retinaface/DATA/Images'
    json_path = 'F:/Leisure_Projects/retinaface/DATA/Images.json'
    images_2_json(image_dir, json_path)