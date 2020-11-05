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
            img, boxes = aug(img, boxes, input_shape)
            # for obj in range(boxes.shape[0]):
            #     xc, yc, w, h = boxes[obj][:4]
            #     cv2.rectangle(img, (int(input_shape[1]*(xc-w/2.)),int(input_shape[0]*(yc-h/2.))), (int(input_shape[1]*(xc+w/2.)),int(input_shape[0]*(yc+h/2.))), 1, 2)
            # cv2.imshow("tmp1", img)
            # cv2.waitKey(0)
            img = np.expand_dims(img, axis=-1)
            image_batch[i] = img
            if boxes.shape[0] <= 0:
                continue

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


def aug(img, boxes, input_shape):
    labels = boxes[:,4:5]
    boxes = boxes[:,:4]
    h, w = input_shape

    # scale & shift & rotate
    if random.uniform(0, 1)>0.5:
        scale_ratio = random.choice([0.25, 0.5, 0.9, 1.0, 1.25])
        new_h = boxes[:, -1] * scale_ratio * h
        if np.min(new_h)<15:
            scale_ratio = 1.
        boxes = boxes * scale_ratio
        if scale_ratio!=1:
            new_h, new_w = int(h*scale_ratio), int(w*scale_ratio)
            img = cv2.resize(img, (new_w, new_h))
            if scale_ratio>1:
                # cut
                gap_h, gap_w = new_h-h, new_w-w
                img = img[gap_h//2:gap_h//2+h, gap_w//2:gap_w//2+w]
                boxes_xcyc = boxes[:,:2] - [gap_w/2./w, gap_h/2./h]
            if scale_ratio<1:
                # pad
                pad_h, pad_w = h-new_h, w-new_w
                img = cv2.copyMakeBorder(img, top=pad_h//2, bottom=pad_h-pad_h//2,
                                         left=pad_w//2, right=pad_w-pad_w//2,
                                         borderType=cv2.BORDER_CONSTANT, value=0)
                boxes_xcyc = boxes[:,:2] + [pad_w/2./w, pad_h/2./h]
            boxes_wh = boxes[:,2:]
            boxes = np.concatenate([boxes_xcyc, boxes_wh], axis=-1)

    if random.uniform(0, 1)>0.5:
        shift_h = random.randint(-20, 20)
        shift_w = random.randint(-20, 20)
        img = cv2.copyMakeBorder(img, top=40, bottom=40, left=40, right=40,
                                 borderType=cv2.BORDER_CONSTANT, value=0)
        start_h, start_w = 40+shift_h, 40+shift_w
        img = img[start_h:start_h+h, start_w:start_w+w]
        boxes_xcyc = boxes[:,:2] - [shift_w/w, shift_h/h]
        boxes_wh = boxes[:,2:]
        boxes = np.concatenate([boxes_xcyc, boxes_wh], axis=-1)

    if random.uniform(0, 1)>0.5:
        rotate_angle = random.uniform(-math.pi/18, math.pi/18)
        tl = boxes[:,:2]*[w,h] + np.stack([-boxes[:,2]*w/2, -boxes[:,3]*h/2], axis=-1)
        tr = boxes[:,:2]*[w,h] + np.stack([boxes[:,2]*w/2, -boxes[:,3]*h/2], axis=-1)
        bl = boxes[:,:2]*[w,h] + np.stack([-boxes[:,2]*w/2, boxes[:,3]*h/2], axis=-1)
        br = boxes[:,:2]*[w,h] + np.stack([boxes[:,2]*w/2, boxes[:,3]*h/2], axis=-1)
        n_points = tl.shape[0]
        points = list(tl) + list(tr) + list(bl) + list(br)
        img, points = rotate_img(rotate_angle, img, points)
        new_tl = np.array(points[:n_points])
        new_tr = np.array(points[n_points:n_points*2])
        new_bl = np.array(points[n_points*2:n_points*3])
        new_br = np.array(points[n_points*3:])
        left = np.minimum(new_tl[:,0], new_bl[:,0])
        right = np.maximum(new_tr[:,0], new_br[:,0])
        top = np.minimum(new_tl[:,1], new_tr[:,1])
        bottom = np.maximum(new_bl[:,1], new_br[:,1])
        xc = (left + right) / 2 / w
        yc = (top + bottom) / 2 / h
        w = np.abs(right - left) / w
        h = np.abs(bottom - top) / h
        boxes = np.stack([xc, yc, w, h], axis=-1)

    aug_boxes = np.concatenate([boxes, labels], axis=-1)
    boxes = valid_boxes(aug_boxes)

    return img, boxes


def valid_boxes(boxes):
    valid_boxes = []
    left = boxes[:, 0] - 0.5*boxes[:,2]
    right = boxes[:, 0] + 0.5*boxes[:,2]
    top = boxes[:, 1] - 0.5*boxes[:,3]
    bottom = boxes[:, 1] + 0.5*boxes[:,3]

    for i in range(boxes.shape[0]):
        if 0<top[i]<1 and 0<bottom[i]<1 and 0<left[i]<1 and 0<right[i]<1:
            valid_boxes.append(boxes[i])
    return np.array(valid_boxes)


def rotate_img(angle, img, points=[], interpolation=cv2.INTER_LINEAR):
    h, w = img.shape
    rotataMat = cv2.getRotationMatrix2D((w/2, h/2), math.degrees(angle), 1)
    # img
    rotate_img = cv2.warpAffine(img, rotataMat, (w, h), flags=interpolation, borderValue=(0,0,0))
    # points
    rotated_points = []
    for point in points:
        point = rotataMat.dot([[point[0]], [point[1]], [1]])
        rotated_points.append((int(point[0]), int(point[1])))
    return rotate_img, rotated_points


if __name__ == '__main__':

    # anchors = get_anchors(anchors, strides=8)
    # print(anchors)

    data_dir = "data/"
    batch_size = 1
    input_shape = (480, 640)     # hw
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
            cls_prob = np.max(gt[:,:,:,4:], axis=-1)
            coords = np.where(cls_prob>0.5)
            print("strides: ", s, "n_objects: ", len(coords[0]))

            img = image_batch[0, :,:,0]

            if len(coords[0]):
                # vis
                for obj in range(len(coords[0])):
                    h,w,a = coords[0][obj], coords[1][obj], coords[2][obj]
                    box = gt[h,w,a]    # normed-rela-origin [xc, yc, w, h]
                    print(box)
                    xc, yc, w, h = box[:4]
                    cv2.rectangle(img, (int(640*(xc-w/2.)), int(480*(yc-h/2.))), (int(640*(xc+w/2.)), int(480*(yc+h/2.))), 255, 2)

                cv2.imshow("tmp", img)
                cv2.waitKey(0)

        if idx>100:
            break






