import keras.backend as K
import tensorflow as tf
import numpy as np
import math


def det_loss(args, n_anchors, n_classes):
    # *y_true: [B,H,W,a,c+4] for each resolution, start by x8level3
    # *cls_pred: [B,H,W,a,c] for each resolution
    # *box_pred: [B,H,W,a,4] for each resolution
    n_levels = len(args) // 3
    cls_preds = args[:n_levels]
    box_pres = args[n_levels:n_levels*2]
    y_true = args[n_levels*2:]
    cls_losses = []
    box_losses = []
    for i in range(n_levels):
        cls_losses.append(focal(y_true[i][...,:n_classes+1], cls_preds[i]))
        box_losses.append(smooth_l1(y_true[i][...,n_classes:], box_pres[i], i))
    cls_loss = tf.add_n(cls_losses)
    box_loss = tf.add_n(box_losses)
    det_loss = box_loss + cls_loss

    return det_loss


# for every resolution
def focal(y_true, y_pred, alpha=0.25, gamma=2.0, cutoff=0.5):
    # y_true: [B,H,W,a,c+1], last dim refers to anchor state {-1:ignore, 0:negative, 1:positive}
    # y_pred: [B,H,W,a,c], logits
    cls_true = y_true[...,:-1]
    anchor_state = y_true[...,-1]      # [B,H,W,a]

    # valid anchors
    indices = tf.where(tf.not_equal(anchor_state, -1))
    cls_true = tf.gather_nd(cls_true, indices)     # [N,c]
    cls_pred = tf.gather_nd(y_pred, indices)
    cls_pred = K.softmax(y_pred)     # posibility

    # normalizer
    normalizer = tf.cast(K.maximum(1, K.shape(cls_true)[0]), tf.float32)

    # weighting
    cls_pt = tf.where(cls_true>cutoff, cls_pred, 1-cls_pred)
    alpha_factor = tf.where(cls_true>cutoff, tf.ones_like(cls_true)*alpha, tf.ones_like(cls_true)*(1-alpha))
    focal_factor = K.pow(1-cls_pt, gamma)

    loss = -alpha_factor*focal_factor*K.log(cls_pt)
    loss = K.sum(loss) / normalizer

    return loss


# for every resolution
def smooth_l1(y_true, y_pred, level, sigma=3.0):
    # y_true: [B,H,W,a,1+4], first dim refers to anchor state {-1:ignore, 0:negative, 1:positive}
    # y_pred: [B,H,W,a,4]
    box_true = y_true[...,:-1]
    box_true = abs2offset(box_true, level)
    anchor_state = y_true[...,-1]

    # valid anchors
    indices = tf.where(tf.equal(anchor_state, 1))
    box_true = tf.gather_nd(box_true, indices)      # [N,4]
    box_pred = tf.gather_nd(y_pred, indices)

    # normalizer
    normalizer = tf.cast(K.maximum(1, K.shape(indices)[0]), tf.float32)

    l1 = K.abs(box_pred-box_true)
    loss = tf.where(K.less(l1, 1./sigma**2), 0.5*(sigma*l1)**2, l1-0.5/sigma**2)
    loss = K.sum(loss) / normalizer

    return loss


anchor_config = {
    'anchor_ratios': [2,1,0.5],
    'anchor_scales': [math.pow(2,0), math.pow(2,1/3), math.pow(2,2/3)],
    'anchor_size': [32,64,128,256,512],
    'output_strides': [8,16,32,64,128]
}


def get_anchors(anchor_config, level):
    size = anchor_config['anchor_size'][level]
    anchors = []
    for scale in anchor_config['anchor_scales']:
        area = size * size * scale
        for ratio in anchor_config['anchor_ratios']:
            w = int(math.sqrt(area/ratio))
            h = int(w * ratio)
            anchors.append([h,w])
    return np.array(anchors), anchor_config['output_strides'][level]


# convert abs box position to offset based on each anchor
def abs2offset(yt_boxes, level, anchor_config=anchor_config):
    # yt_boxes: [B,H,W,a,4], abs [y_min, x_min, y_max, x_max]
    # offset_boxes: [B,H,W,a,4], offset [tx,ty,tw,th]
    # anchor
    anchors, strides = get_anchors(anchor_config, level)     # [9,2] [h,w]
    n_anchors = anchors.shape[0]
    anchors = K.cast(K.reshape(anchors, (-1,1,1,n_anchors,2)), tf.float32)
    # tw, th
    yt_h = yt_boxes[...,2] - yt_boxes[...,0]
    offset_h = K.log(yt_h / anchors[...,0])
    yt_w = yt_boxes[...,3] - yt_boxes[...,1]
    offset_w = K.log(yt_w / anchors[...,1])

    # grid
    grid_shape = K.int_shape(yt_boxes)[1:3]
    h, w = grid_shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    anchors_x = K.cast(K.reshape(x, (1,h,w,1)) * strides, tf.float32)
    anchors_y = K.cast(K.reshape(y, (1,h,w,1)) * strides, tf.float32)

    # ty, tx
    yt_center_y = (yt_boxes[...,0] + yt_boxes[...,2]) / 2.
    offset_y = (yt_center_y - anchors_y) / anchors[...,0]
    yt_center_x = (yt_boxes[...,1] + yt_boxes[...,3]) / 2.
    offset_x = (yt_center_x - anchors_x) / anchors[...,1]

    offset_boxes = K.stack([offset_x,offset_y,offset_w,offset_h], axis=-1)
    return offset_boxes

