import cv2

import numpy as np
import mxnet as mx

from . import face_preprocess

from sklearn import preprocessing as preprocessing

class InsightFace:

    def __init__(self, prefix, epoch, ctx_id=0, image_size=(112,112)):

        self.ctx_id = ctx_id
        self.image_size = image_size

        if self.ctx_id >= 0:
            self.ctx = mx.gpu(self.ctx_id)
        else:
            self.ctx = mx.cpu()

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']

        self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        self.model.bind(data_shapes=[('data', (1, 3, self.image_size[0], self.image_size[1]))])
        self.model.set_params(arg_params, aux_params)

    @staticmethod
    def prepare_insight_input(face, landmark, face_img):

        box = face[:4]
        nimg = face_preprocess.preprocess(face_img, box, landmark, image_size='112,112')

        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))

        return aligned

    def face_vertorizing(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        vector = self.model.get_outputs()[0].asnumpy()
        vector = preprocessing.normalize(vector).flatten()
        return vector

    @staticmethod
    def vector_diff(v1, v2):
        return np.sum(np.square(v1 - v2))