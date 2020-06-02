import keras.backend as K
import tensorflow as tf
import numpy as np
import math


anchor_params = {
    'sizes': [32, 64, 128, 256, 512],
    'strides': [8, 16, 32, 64, 128],
    'ratios': [0.5, 1, 2],
    'scales': [2**0, 2**(1.0/3.0), 2**(2.0/3.0)]
}


# raw linear outputs to box
def bbox(y_preds, input_shape, n_classes, anchor_params=anchor_params,
         nms_threshold=0.5, score_threshold=0.5, max_detections=300):
    # anchors: axis1-concat [H*W*9, 4], coords[y_min, x_min, y_max, x_max]
    anchors = generate_anchors(input_shape, anchor_params)
    x_a = (anchors[...,1] + anchors[...,3]) / 2.     # [H*W*9, 1]
    y_a = (anchors[...,0] + anchors[...,2]) / 2.
    w_a = anchors[...,3] - anchors[...,1] + 1.
    h_a = anchors[...,0] - anchors[...,2] + 1.
    # preds: offsets output[B,H,W,4*9/c*9], coords[y_min, x_min, y_max, x_max], reflect back + clip
    n_levels = len(anchor_params['strides'])
    cls_preds = y_preds[:n_levels]
    cls_preds = cls_preds.reshape((-1, n_classes))
    box_preds = y_preds[n_levels:]
    box_predoff_ymin = box_preds[..., 0::4].reshape((-1,1))    # [B,H*W*9]
    box_predoff_xmin = box_preds[..., 1::4].reshape((-1,1))
    box_predoff_ymax = box_preds[..., 2::4].reshape((-1,1))
    box_predoff_xmax = box_preds[..., 4::4].reshape((-1,1))
    box_pred_ymin = K.maximum(0., box_predoff_ymin * h_a + y_a)
    box_pred_xmin = K.maximum(0., box_predoff_xmin * w_a + x_a)
    box_pred_ymax = K.minimum(input_shape[0]-1, box_predoff_ymax * h_a + y_a)
    box_pred_xmin = K.minimum(input_shape[1]-1, box_predoff_xmax * w_a + x_a)
    box_preds = tf.concat([box_pred_ymin, box_pred_xmin, box_pred_ymax, box_pred_xmin, cls_preds], axis=-1)
    # nms
    boxes, scores, classes = nms(box_preds, n_classes, nms_threshold, score_threshold, max_detections)

    return boxes, scores, classes


def generate_anchors(input_shape, anchor_params):
    # generator anchors without anchor state for inference, [H,W,9*4] -> [H*W*9,4]
    # concat each level
    anchors = []
    for i in range(len(anchor_params['strides'])):   # n_levels
        coords_y = [i for i in range(input_shape[0]//anchor_params['strides'][i])]
        coords_x = [i for i in range(input_shape[1]//anchor_params['strides'][i])]
        grid_x, grid_y = np.meshgrid(coords_x, coords_y)
        ctr_y, ctr_x = grid_y+0.5, grid_x+0.5
        for s in range(len(anchor_params['scales'])):
            for r in range(len(anchor_params['ratios'])):
                abs_h = anchor_params['sizes'][i] * anchor_params['scales'][s] * math.sqrt(anchor_params['ratios'][r])
                abs_w = anchor_params['sizes'][i] * anchor_params['scales'][s] / math.sqrt(anchor_params['ratios'][r])
                abs_cy = ctr_y * anchor_params['strides'][i]
                abs_cx = ctr_x * anchor_params['strides'][i]
                abs_y_min = np.maximum(0, (abs_cy - abs_h/2.).astype(np.int)).reshape((-1))
                abs_x_min = np.maximum(0, (abs_cx - abs_w/2.).astype(np.int)).reshape((-1))
                abs_y_max = np.minimum(input_shape[0]-1, (abs_cy - abs_h/2.).astype(np.int)).reshape((-1))
                abs_x_max = np.minimum(input_shape[1]-1, (abs_cx - abs_w/2.).astype(np.int)).reshape((-1))
                anchor_coords = np.stack([abs_y_min, abs_x_min, abs_y_max, abs_x_max], axis=-1)
                anchors.append(anchor_coords)
    return np.concatenate(anchors, axis=0)


def nms(box_preds, n_classes, nms_threshold=0.5, score_threshold=0.5, max_detections=300):
    # box_preds: [N,4+c], coords+p
    # with/without object
    mask = tf.where(box_preds[..., -n_classes:]>score_threshold)

    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(n_classes):
        cls_idx = tf.where(box_preds[..., 4+c]>score_threshold)
        class_boxes = K.gather(box_preds[...,:4], cls_idx)
        class_box_scores = K.gather(box_preds[...,:4+c], cls_idx)
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_detections, iou_threshold=nms_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)      # [N,4]
    scores_ = K.concatenate(scores_, axis=0)    # [N,]
    classes_ = K.concatenate(classes_, axis=0)  # [N,]

    return boxes_, scores_, classes_      # coords+p+cls


if __name__ == '__main__':

    anchors = generate_anchors((640,512), anchor_params)
    print(anchors.shape)




