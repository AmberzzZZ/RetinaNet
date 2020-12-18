import cv2
import numpy as np
from keras.layers import Input, Lambda, concatenate
from keras.models import Model
import keras.backend as K
from dataGenerator import anchors, get_anchors  #, get_anchors_yolo
from retinaNet import retinaNet
import sys
sys.path.append('/Users/amber/workspace/nms')
from hard_nms import hard_nms
from soft_nms import soft_nms
from diou_nms import diou_nms
from fast_nms import fast_nms
from cluster_nms import cluster_nms


def sigmoid(x):
    return 1 / (1 + np.exp(1e-9-x))


# def write_xml()


if __name__ == '__main__':

    data_dir = "data/"

    # base model
    num_classes = 2
    input_shape = (512, 512)     # hw
    num_classes = 6
    strides = [8,16,32,64,128]
    anchors = get_anchors(anchors, strides)
    # anchors = get_anchors_yolo('anchors.txt', n_anchors)
    n_anchors = anchors[0].shape[0]
    score_thresh = 0.2

    # model
    model = retinaNet(input_shape=input_shape+(1,), backbone='resnet', depth=50, fpn_filters=256,
                      n_classes=num_classes, n_anchors=n_anchors, anchors=anchors, strides=strides)
    # model.load_weights()
    # model.summary()

    # add postprocess tail: sigmoid
    reshape_idx = 1
    cls_ouputs = ['reshape_%s' % str(i+reshape_idx) for i in range(len(strides))]
    cls_ouputs = [model.get_layer(name=n).output for n in cls_ouputs]
    box_ouputs = ['reshape_%s' % str(i+len(strides)+reshape_idx) for i in range(len(strides))]
    box_ouputs = [model.get_layer(name=n).output for n in box_ouputs]
    outputs = [concatenate([i,j]) for i,j in zip(box_ouputs, cls_ouputs)]
    model = Model(model.input[0], outputs)

    # test
    img = cv2.imread("data/tux_hacking.png", 1)
    if np.max(img)>1:
        img = img / 255.
    inpt = np.expand_dims(img, axis=0)

    # run model
    # outputs = model.predict(inpt)
    outputs = [[] for i in range(len(strides))]
    boxes_filtered = []
    scores_filtered = []
    ids_filtered = []
    for i, s in enumerate(strides):
        grid_h, grid_w = input_shape[0]//s, input_shape[1]//s
        anchors_s = anchors[i]
        # test with fake outputs
        output_xy = np.random.uniform(-0.5, 0.5, (grid_h,grid_w,n_anchors,2))
        output_wh = np.random.uniform(-10000, 10000, (grid_h,grid_w,n_anchors,2))
        output_cls = np.random.uniform(0,0.6,(grid_h,grid_w,n_anchors,num_classes))
        output = np.concatenate([output_xy, output_wh, output_cls], axis=-1)

        # labels & probs
        cls = np.argmax(output[:,:,:,4:], axis=-1)
        prob = np.max(output[:,:,:,4:], axis=-1)

        # grid offset
        grid_offset_x, grid_offset_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        grid_coords = np.stack([grid_offset_x, grid_offset_y], axis=-1)
        grid_coords = np.expand_dims(grid_coords, axis=2)
        center_coords = grid_coords + 0.5

        # pred_xcycwh_abs
        input_shape = np.array(input_shape)[::-1]
        pred_xcyc_abs = output[:,:,:,:2] * anchors_s + (center_coords/[grid_w, grid_h]*input_shape)
        pred_wh_abs = np.exp(output[:,:,:,:2]) * anchors_s
        # pred_x1y1x2y2_abs
        pred_x1y1 = pred_xcyc_abs - pred_wh_abs/2
        pred_x2y2 = pred_xcyc_abs + pred_wh_abs/2
        boxes = np.concatenate([pred_x1y1, pred_x2y2], axis=-1).reshape((-1,4))
        scores = np.reshape(prob, (-1,1))
        labels = np.reshape(cls, (-1,1))

        # nms
        boxes_, scores_, labels_ = cluster_nms(boxes, scores, labels,
                                               score_thresh=0.3, iou_thresh=0.2, max_detections=100)
        print("stride %d filtered boxes: " % s, boxes_.shape, scores_.shape, labels_.shape)
        boxes_ = boxes_.astype(np.int)

        boxes_filtered.append(boxes_)
        scores_filtered.append(scores_)
        ids_filtered.append(labels_)

    boxes_filtered = np.concatenate(boxes_filtered, axis=0)
    print(boxes_filtered.shape)
    scores_filtered = np.concatenate(scores_filtered, axis=0)
    ids_filtered = np.concatenate(ids_filtered, axis=0)
    print(np.unique(ids_filtered))














