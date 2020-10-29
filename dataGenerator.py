import numpy as np
import cv2
import random
import math
import os


anchors = {'scale': [2**0, 2**(1/3.), 2**(2/3.)],
           'ratio': [0.5, 1, 2],
           'size': {8:32, 16:64, 32:128, 64:256, 128:512}}


def data_generator(data_dir, batch_size, input_shape, num_classes, anchors, strides=[8,16,32]):
    # img: [b,h,w,1], y_true: multi-scales of [b,h,w,a,4+c]
    full_lst = [i.split('.txt')[0] for i in os.listdir(data_dir) if 'txt' in i]
    n_anchors = len(anchors['scale']) * len(anchors['ratio'])
    anchors = get_anchors(anchors, strides)   # list of arr, [9,2] for each level

    while 1:
        random.shuffle(full_lst)
        image_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 1))
        y_true = [np.zeros((batch_size, input_shape[0]//s, input_shape[1]//s,
                           n_anchors, 4+num_classes+1)) for s in strides]
        for i in range(batch_size):
            file_name = full_lst[i]
            img = cv2.imread(os.path.join(data_dir, file_name+'.jpg'), 0)
            if np.max(img)>1:
                img = img / 255.
            boxes = get_box(os.path.join(data_dir, file_name+'.txt'))
            # img, boxes = aug(img, boxes)

            img = np.expand_dims(img, axis=-1)
            image_batch[i] = img
            input_shape = np.array(input_shape)
            boxes_xy = boxes[...,:2] * input_shape[::-1]
            boxes_wh = boxes[...,2:4] * input_shape[::-1]
            # tranverse strides, generate y_true
            for idx, s in enumerate(strides):
                anchors_s = np.expand_dims(anchors[idx], axis=0)
                anchor_maxes = anchors_s / 2.
                anchor_mins = -anchor_maxes

                box_maxes = boxes_wh / 2.
                box_mins = -box_maxes
                # center shift
                grid_xy = np.floor(boxes_xy/s)
                center_shift_xy = boxes_xy - (grid_xy+0.5)*s
                box_maxes += center_shift_xy
                box_mins += center_shift_xy
                box_maxes = np.expand_dims(box_maxes, axis=1)
                box_mins = np.expand_dims(box_mins, axis=1)

                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
                box_area = boxes_wh[..., 0:1] * boxes_wh[..., 1:2]
                anchor_area = anchors_s[..., 0] * anchors_s[..., 1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                best_anchor = np.argmax(iou, axis=-1)

                for box_idx, anchor_idx in enumerate(best_anchor):
                    # print(iou[box_idx, anchor_idx])
                    if iou[box_idx, anchor_idx] < 0.5:
                        continue
                    grid_w, grid_h = map(int, grid_xy[box_idx])
                    if y_true[idx][i, grid_h,grid_w,anchor_idx,-1] > iou[box_idx, anchor_idx]:
                        continue
                    y_true[idx][i, grid_h,grid_w,anchor_idx,:4] = boxes[box_idx,:4]
                    cls_id = int(boxes[box_idx][-1])
                    y_true[idx][i, grid_h,grid_w,anchor_idx,4+cls_id] = 1
                    y_true[idx][i, grid_h,grid_w,anchor_idx,-1] = iou[box_idx, anchor_idx]

        y_true = [i[...,:-1] for i in y_true]

        yield [image_batch, *y_true], np.zeros(batch_size)


def get_anchors(anchors, strides):
    anchor_ratios = anchors['ratio']
    anchor_scales = anchors['scale']
    anchor_sizes = anchors['size']

    anchors = []
    for s in strides:
        base_size = anchor_sizes[s]
        anchors_s = base_size * np.tile(np.expand_dims(anchor_scales,axis=-1), (len(anchor_ratios),2))
        anchors_s[:,0] = anchors_s[:,0] / np.sqrt(np.repeat(anchor_ratios, len(anchor_scales)))
        anchors_s[:,1] = anchors_s[:,0] * np.repeat(anchor_ratios, len(anchor_scales))
        anchors.append(anchors_s)

    return anchors


def get_box(yolo_file):
    f = open(yolo_file, 'r')
    boxes = []
    for line in f.readlines():
        if len(line) < 5:
            continue
        cls, xc, yc, w, h = map(float, line.strip().split(' '))
        boxes.append([xc, yc, w, h, cls])
    return np.array(boxes)


if __name__ == '__main__':

    # anchors = get_anchors(anchors, strides=8)
    # print(anchors)

    data_dir = "data/"
    batch_size = 1
    input_shape = (480, 640)
    num_classes = 2
    strides = [8,16,32,64,128]
    train_generator = data_generator(data_dir, batch_size, input_shape, num_classes, anchors, strides=strides)

    for idx, [x_batch, _] in enumerate(train_generator):
        image_batch = x_batch[0]
        print("image_batch: ", image_batch.shape, np.max(image_batch))
        y_true = x_batch[1:]
        print("y_true: ", [i.shape for i in y_true])

        for idx, s in enumerate(strides):
            gt = y_true[idx][0]
            coords = np.where(gt[:,:,:,0]>0)
            print(coords)

            if s==64:
                # vis
                img = cv2.imread("data/tux_hacking.jpg", 0)
                box1 = gt[2,5,3]    # normed-rela-origin [xc, yc, w, h]
                print(box1)
                xc, yc, w, h = box1[:4]
                cv2.rectangle(img, (int(640*(xc-w/2.)), int(480*(yc-h/2.))), (int(640*(xc+w/2.)), int(480*(yc+h/2.))), 255, 2)

                box2 = gt[5,4,0]
                print(box2)
                xc, yc, w, h = box2[:4]
                cv2.rectangle(img, (int(640*(xc-w/2.)), int(480*(yc-h/2.))), (int(640*(xc+w/2.)), int(480*(yc+h/2.))), 255, 2)

                cv2.imshow("tmp", img)
                cv2.waitKey(0)

        if idx>0:
            break






