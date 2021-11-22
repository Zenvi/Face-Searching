from __future__ import print_function
import datetime
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper

class RetinaFace:
    def __init__(self,
                 prefix,
                 epoch,
                 ctx_id=0,
                 nms=0.4,
                 decay4=0.5,
                 vote=False):
        self.ctx_id = ctx_id
        self.decay4 = decay4
        self.nms_threshold = nms
        self.vote = vote
        self.fpn_keys = []
        self.anchor_cfg = None
        pixel_means = [0.0, 0.0, 0.0]
        pixel_stds = [1.0, 1.0, 1.0]
        pixel_scale = 1.0
        self.preprocess = False
        _ratio = (1., )

        self._feat_stride_fpn = [32, 16, 8]
        self.anchor_cfg = {
                           '32': {'SCALES': (32, 16),
                                  'BASE_SIZE': 16,
                                  'RATIOS': _ratio,
                                  'ALLOWED_BORDER': 9999},
                           '16': {'SCALES': (8, 4),
                                  'BASE_SIZE': 16,
                                  'RATIOS': _ratio,
                                  'ALLOWED_BORDER': 9999},
                           '8': {'SCALES': (2, 1),
                                 'BASE_SIZE': 16,
                                 'RATIOS': _ratio,
                                 'ALLOWED_BORDER': 9999},
                           }

        # print(self._feat_stride_fpn, self.anchor_cfg)

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        dense_anchor = False
        self._anchors_fpn = dict(
            zip(self.fpn_keys,
                generate_anchors_fpn(dense_anchor=dense_anchor, cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v

        self._num_anchors = dict(
            zip(self.fpn_keys,
                [anchors.shape[0] for anchors in self._anchors_fpn.values()]))

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        if self.ctx_id >= 0:
            self.ctx = mx.gpu(self.ctx_id)
            self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
        else:
            self.ctx = mx.cpu()
            self.nms = cpu_nms_wrapper(self.nms_threshold)
        self.pixel_means = np.array(pixel_means, dtype=np.float32)
        self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
        self.pixel_scale = float(pixel_scale)
        # print('means', self.pixel_means)
        self.use_landmarks = False
        if len(sym) // len(self._feat_stride_fpn) >= 3:
            self.use_landmarks = True
        # print('use_landmarks', self.use_landmarks)
        self.cascade = 0
        if float(len(sym)) // len(self._feat_stride_fpn) > 3.0:
            self.cascade = 1
        # print('cascade', self.cascade)
        self.bbox_stds = [1.0, 1.0, 1.0, 1.0]
        self.landmark_std = 1.0

        # print('sym size:', len(sym))

        image_size = (640, 640)
        self.model = mx.mod.Module(symbol=sym,
                                   context=self.ctx,
                                   label_names=None)
        self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))],
                        for_training=False)
        self.model.set_params(arg_params, aux_params)

    def get_input(self, img):
        im = img.astype(np.float32)
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / self.pixel_stds[2 - i]
        data = nd.array(im_tensor)
        return data

    def detect(self, img, threshold=0.5, scales=1.0):

        proposals_list = []
        scores_list = []
        landmarks_list = []
        strides_list = []
        timea = datetime.datetime.now()

        if scales != 1.0:
            im = cv2.resize(img,
                            None,
                            None,
                            fx=scales,
                            fy=scales,
                            interpolation=cv2.INTER_LINEAR)
        else:
            im = img.copy()

        im = im.astype(np.float32)

        im_info = [im.shape[0], im.shape[1]]
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))

        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / self.pixel_stds[2 - i]

        data = nd.array(im_tensor)
        db = mx.io.DataBatch(data=(data, ),
                             provide_data=[('data', data.shape)])

        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()

        sym_idx = 0

        for _idx, s in enumerate(self._feat_stride_fpn):

            _key = 'stride%s' % s
            stride = int(s)
            is_cascade = False
            if self.cascade:
                is_cascade = True

            scores = net_out[sym_idx].asnumpy()
            scores = scores[:, self._num_anchors['stride%s' % s]:, :, :]
            bbox_deltas = net_out[sym_idx + 1].asnumpy()
            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            A = self._num_anchors['stride%s' % s]
            K = height * width
            anchors_fpn = self._anchors_fpn['stride%s' % s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)

            anchors = anchors.reshape((K * A, 4))

            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3] // A

            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * self.bbox_stds[0]
            bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * self.bbox_stds[1]
            bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * self.bbox_stds[2]
            bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * self.bbox_stds[3]
            proposals = self.bbox_pred(anchors, bbox_deltas)

            # if is_cascade:
            #     cascade_sym_num = 0
            #     cls_cascade = False
            #     bbox_cascade = False
            #     __idx = [3, 4]
            #     if not self.use_landmarks:
            #         __idx = [2, 3]
            #     for diff_idx in __idx:
            #         if sym_idx + diff_idx >= len(net_out):
            #             break
            #         body = net_out[sym_idx + diff_idx].asnumpy()
            #         if body.shape[1] // A == 2:  #cls branch
            #             if cls_cascade or bbox_cascade:
            #                 break
            #             else:
            #                 cascade_scores = body[:, self.
            #                                       _num_anchors[
            #                                           'stride%s' %
            #                                           s]:, :, :]
            #                 cascade_scores = cascade_scores.transpose(
            #                     (0, 2, 3, 1)).reshape((-1, 1))
            #                 #scores = (scores+cascade_scores)/2.0
            #                 scores = cascade_scores  #TODO?
            #                 cascade_sym_num += 1
            #                 cls_cascade = True
            #                 #print('find cascade cls at stride', stride)
            #         elif body.shape[1] // A == 4:  #bbox branch
            #             cascade_deltas = body.transpose(
            #                 (0, 2, 3, 1)).reshape(
            #                     (-1, bbox_pred_len))
            #             cascade_deltas[:, 0::
            #                            4] = cascade_deltas[:, 0::
            #                                                4] * self.bbox_stds[
            #                                                    0]
            #             cascade_deltas[:, 1::
            #                            4] = cascade_deltas[:, 1::
            #                                                4] * self.bbox_stds[
            #                                                    1]
            #             cascade_deltas[:, 2::
            #                            4] = cascade_deltas[:, 2::
            #                                                4] * self.bbox_stds[
            #                                                    2]
            #             cascade_deltas[:, 3::
            #                            4] = cascade_deltas[:, 3::
            #                                                4] * self.bbox_stds[
            #                                                    3]
            #             proposals = self.bbox_pred(
            #                 proposals, cascade_deltas)
            #             cascade_sym_num += 1
            #             bbox_cascade = True
            #             #print('find cascade bbox at stride', stride)

            proposals = clip_boxes(proposals, im_info[:2])

            if stride == 4 and self.decay4 < 1.0:
                scores *= self.decay4

            scores_ravel = scores.ravel()

            order = np.where(scores_ravel >= threshold)[0]

            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:, 0:4] /= scales

            proposals_list.append(proposals)
            scores_list.append(scores)
            if self.nms_threshold < 0.0:
                _strides = np.empty(shape=(scores.shape), dtype=np.float32)
                _strides.fill(stride)
                strides_list.append(_strides)

            if not self.vote and self.use_landmarks:
                landmark_deltas = net_out[sym_idx + 2].asnumpy()
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
                landmark_deltas *= self.landmark_std
                landmarks = self.landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                landmarks[:, :, 0:2] /= scales
                landmarks_list.append(landmarks)

            if self.use_landmarks:
                sym_idx += 3
            else:
                sym_idx += 2
            # if is_cascade:
            #     sym_idx += cascade_sym_num

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if self.use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            if self.nms_threshold < 0.0:
                return np.zeros((0, 6)), landmarks
            else:
                return np.zeros((0, 5)), landmarks
        scores = np.vstack(scores_list)

        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]
        if self.nms_threshold < 0.0:
            strides = np.vstack(strides_list)
            strides = strides[order]
        if not self.vote and self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        if self.nms_threshold > 0.0:
            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            if not self.vote:
                keep = self.nms(pre_det)
                det = np.hstack((pre_det, proposals[:, 4:]))
                det = det[keep, :]
                if self.use_landmarks:
                    landmarks = landmarks[keep]
            else:
                det = np.hstack((pre_det, proposals[:, 4:]))
                det = self.bbox_vote(det)
        elif self.nms_threshold < 0.0:
            det = np.hstack((proposals[:, 0:4], scores, strides)).astype(np.float32, copy=False)
        else:
            det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)

        return det, landmarks

    def detect_center(self, img, threshold=0.5, scales=1.0):
        det, landmarks = self.detect(img, threshold, scales)
        if det.shape[0] == 0:
            return None, None
        bindex = 0
        if det.shape[0] > 1:
            img_size = np.asarray(img.shape)[0:2]
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
        bbox = det[bindex, :]
        landmark = landmarks[bindex, :, :]
        return bbox, landmark

    @staticmethod
    def check_large_pose(landmark, bbox):
        assert landmark.shape == (5, 2)
        assert len(bbox) == 4

        def get_theta(base, x, y):
            vx = x - base
            vy = y - base
            vx[1] *= -1
            vy[1] *= -1
            tx = np.arctan2(vx[1], vx[0])
            ty = np.arctan2(vy[1], vy[0])
            d = ty - tx
            d = np.degrees(d)
            if d < -180.0:
                d += 360.
            elif d > 180.0:
                d -= 360.0
            return d

        landmark = landmark.astype(np.float32)

        theta1 = get_theta(landmark[0], landmark[3], landmark[2])
        theta2 = get_theta(landmark[1], landmark[2], landmark[4])
        theta3 = get_theta(landmark[0], landmark[2], landmark[1])
        theta4 = get_theta(landmark[1], landmark[0], landmark[2])
        theta5 = get_theta(landmark[3], landmark[4], landmark[2])
        theta6 = get_theta(landmark[4], landmark[2], landmark[3])
        theta7 = get_theta(landmark[3], landmark[2], landmark[0])
        theta8 = get_theta(landmark[4], landmark[1], landmark[2])

        left_score = 0.0
        right_score = 0.0
        up_score = 0.0
        down_score = 0.0
        if theta1 <= 0.0:
            left_score = 10.0
        elif theta2 <= 0.0:
            right_score = 10.0
        else:
            left_score = theta2 / theta1
            right_score = theta1 / theta2
        if theta3 <= 10.0 or theta4 <= 10.0:
            up_score = 10.0
        else:
            up_score = max(theta1 / theta3, theta2 / theta4)
        if theta5 <= 10.0 or theta6 <= 10.0:
            down_score = 10.0
        else:
            down_score = max(theta7 / theta5, theta8 / theta6)
        mleft = (landmark[0][0] + landmark[3][0]) / 2
        mright = (landmark[1][0] + landmark[4][0]) / 2
        box_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        ret = 0
        if left_score >= 3.0:
            ret = 1
        if ret == 0 and left_score >= 2.0:
            if mright <= box_center[0]:
                ret = 1
        if ret == 0 and right_score >= 3.0:
            ret = 2
        if ret == 0 and right_score >= 2.0:
            if mleft >= box_center[0]:
                ret = 2
        if ret == 0 and up_score >= 2.0:
            ret = 3
        if ret == 0 and down_score >= 5.0:
            ret = 4
        return ret, left_score, right_score, up_score, down_score

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _filter_boxes2(boxes, max_size, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size > 0:
            keep = np.where(np.minimum(ws, hs) < max_size)[0]
        elif min_size > 0:
            keep = np.where(np.maximum(ws, hs) > min_size)[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        """
      Transform the set of class-agnostic boxes into class-specific boxes
      by applying the predicted offsets (box_deltas)
      :param boxes: !important [N 4]
      :param box_deltas: [N, 4 * num_classes]
      :return: [N 4 * num_classes]
      """
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        # x1
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        # y1
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        # x2
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        # y2
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1] > 4:
            pred_boxes[:, 4:] = box_deltas[:, 4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
            pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
        return pred

    def bbox_vote(self, det):
        if det.shape[0] == 0:
            return np.zeros((0, 5))
        dets = None
        while det.shape[0] > 0:
            if dets is not None and dets.shape[0] >= 750:
                break
            # IOU
            area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[0] + area[:] - inter)

            # nms
            merge_index = np.where(o >= self.nms_threshold)[0]
            det_accu = det[merge_index, :]
            det = np.delete(det, merge_index, 0)
            if merge_index.shape[0] <= 1:
                if det.shape[0] == 0:
                    try:
                        dets = np.row_stack((dets, det_accu))
                    except:
                        dets = det_accu
                continue
            det_accu[:,
                     0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:],
                                                       (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
                det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            if dets is None:
                dets = det_accu_sum
            else:
                dets = np.row_stack((dets, det_accu_sum))
        dets = dets[0:750, :]
        return dets
