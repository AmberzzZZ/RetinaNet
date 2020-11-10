from backbone import resnet, resnext, Conv_BN
from keras.layers import Input, UpSampling2D, add, Conv2D, Lambda, Reshape
from keras.models import Model
import keras.backend as K
from loss import det_loss
from dataGenerator import anchors


backbones = {'resnet': resnet, 'resnext': resnext}


def retinaNet(input_tensor=None, input_shape=(256,256,3),
              backbone='resnet', depth=50, fpn_filters=256, n_classes=10,
              anchors=anchors, strides=[8,16,32,64,128]):
    if input_tensor is not None:
        inpt = input_tensor
        input_shape = inpt._keras_shape[1:]
    else:
        inpt = Input(input_shape)

    # backbone
    x = backbones[backbone](input_shape=input_shape, depth=depth)(inpt)     # [4xC2, 32xC5]

    # fpn
    x = fpn(x[1:], fpn_filters, strides)      # [8xP3, 128xP7]

    # head
    n_anchors = len(anchors['scale']) * len(anchors['ratio'])
    cls_outputs = cls_head(x, fpn_filters, n_classes, n_anchors)
    box_outputs = box_head(x, fpn_filters, n_anchors)

    # y_true
    h, w = input_shape[:2]
    y_true = [Input((h//s, w//s, n_anchors, n_classes+4)) for s in strides]

    # loss
    retina_loss = Lambda(det_loss, arguments={'n_classes': n_classes, 'anchors': anchors, 'strides': strides, 'input_shape': input_shape[:2]})  \
                        ([*cls_outputs,*box_outputs,*y_true])

    # model
    model = Model([inpt, *y_true], retina_loss)

    return model


def fpn(feats, fpn_filters, strides):
    # lateral connections(1x1 conv)
    feats = [Conv_BN(i, fpn_filters, kernel_size=1, strides=1, activation=None) for i in feats]
    C3, C4, C5 = feats
    # top-down connections(upSampling)
    P5 = C5
    P5_up = UpSampling2D(size=2, interpolation='nearest')(P5)
    P4 = add([C4, P5_up])
    P4_up = UpSampling2D(size=2, interpolation='nearest')(P4)
    P3 = add([C3, P4_up])
    # P6 = Conv2D(fpn_filters, 3, strides=2, padding='same')(feats[-1])
    P6 = Conv_BN(feats[-1], fpn_filters, 3, strides=2, activation='relu')
    P7 = Conv_BN(P6, fpn_filters, 3, strides=2, activation='relu')
    feature_dict = {8:P3, 16:P4, 32:P5, 64:P6, 128:P7}
    return [feature_dict[s] for s in strides]


def shared_cls_head(n_filters, n_classes, n_anchors):
    inpt = Input((None,None,n_filters))
    x = inpt
    for i in range(4):
        x = Conv_BN(x, n_filters, 3, strides=1, activation='relu')
    x = Conv2D(n_classes*n_anchors, 3, strides=1, padding='same')(x)
    model = Model(inpt, x, name='shared_cls_head')
    return model


def cls_head(feats, n_filters, n_classes, n_anchors):
    shared_cls_head_submodel = shared_cls_head(n_filters, n_classes, n_anchors)
    cls_outputs = []
    for x in feats:
        x = shared_cls_head_submodel(x)
        h, w = K.int_shape(x)[1:3]
        x = Reshape([h,w,n_anchors,n_classes])(x)
        cls_outputs.append(x)
    return cls_outputs


def shared_box_head(n_filters, n_anchors):
    inpt = Input((None,None,n_filters))
    x = inpt
    for i in range(4):
        x = Conv_BN(x, n_filters, 3, strides=1, activation='relu')
    x = Conv2D(4*n_anchors, 3, strides=1, padding='same')(x)
    model = Model(inpt, x, name='shared_box_head')
    return model


def box_head(feats, n_filters, n_anchors):
    shared_box_head_submodel = shared_box_head(n_filters, n_anchors)
    box_outputs = []
    for x in feats:
        x = shared_box_head_submodel(x)
        h, w = K.int_shape(x)[1:3]
        x = Reshape([h,w,n_anchors,4])(x)
        box_outputs.append(x)
    return box_outputs


if __name__ == '__main__':

    model = retinaNet(input_shape=(256,256,3))
    model.summary()




