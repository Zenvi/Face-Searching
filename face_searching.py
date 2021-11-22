import cv2
import json
import numpy as np
from utils.retinaface import RetinaFace
from utils.insightface import InsightFace

def find_FaceVector(vector, vector_dict):
    dist_thres = 1.0
    min_image_names = []
    for image_name in vector_dict.keys():
        vector_list = vector_dict[image_name]
        for v in vector_list:
            v = np.array(v, dtype=np.float32)
            distance = np.sum(np.square(vector - v))
            if distance < dist_thres:
                min_image_names.append(image_name)
    return min_image_names

# Load the vector dictionary
with open('./DATA/Images.json', 'r') as f:
    vector_dict = json.load(f)

# Load the image which is later used for searching
img = cv2.imread('./TestImages/2.jpg')

# Load the face detection model and face recognition model
print('Loading Models...')
detector = RetinaFace('./model/mnet.25/mnet.25', 0, ctx_id=0)
Recognizer = InsightFace('./model/insightface/insightface', 0, ctx_id=0)
print('Done!')

# Detect faces in the image
faces, landmarks = detector.detect(img, threshold=0.8)

if len(faces) == 1:

    face = faces[0].astype(np.int)
    landmark = landmarks[0].astype(np.int)

    aligned = Recognizer.prepare_insight_input(face, landmark, img)
    img_vector = Recognizer.face_vertorizing(aligned)

    min_image_names = find_FaceVector(img_vector, vector_dict)



    cv2.namedWindow('This is the image you used for searching', 0)
    cv2.imshow('This is the image you used for searching', img)
    cv2.waitKey(0)

    cv2.namedWindow('Search results', 0)
    for name in min_image_names:
        result = cv2.imread('./DATA/Images/' + name)
        cv2.imshow('Search results', result)
        if cv2.waitKey(0) == ord('q'):
            continue

    cv2.destroyAllWindows()

else:
    print('Please input an image with only 1 face.')