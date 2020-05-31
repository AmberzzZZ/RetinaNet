from backbone import resnet, resnext, Conv_BN
from keras.layers import Input, UpSampling2D, add, Conv2D
from keras.models import Model

backbones = {'resnet': resnet, 'resnext': resnext}


def retinaNet(input_tensor=None, input_shape=(224,224,3),
              backbone='resnet', depth=50, fpn_filters=256, n_classes=10, n_anchors=9):
    if input_tensor is not None:
        inpt = input_tensor
        input_shape = inpt._keras_shape[1:]
    else:
        inpt = Input(input_shape)

    # backbone
    x = backbones[backbone](input_shape=input_shape, depth=depth)(inpt)     # [4x, 32x]

    # fpn
    x = fpn(x, fpn_filters)      # [4x, 32x]

    # head
    cls_outputs = cls_head(x, fpn_filters, n_classes, n_anchors)
    box_outputs = box_head(x, fpn_filters, n_anchors)

    # model
    model = Model(inpt, cls_outputs + box_outputs)

    return model


def fpn(feats, fpn_filters):
    # lateral connections(1x1 conv)
    feats = [Conv_BN(i, fpn_filters, kernel_size=1, strides=1, activation=None) for i in feats]
    C2, C3, C4, C5 = feats
    # top-down connections(upSampling)
    P5 = C5
    P5_up = UpSampling2D(size=2, interpolation='nearest')(P5)
    P4 = add([C4, P5_up])
    P4_up = UpSampling2D(size=2, interpolation='nearest')(P4)
    P3 = add([C3, P4_up])
    P3_up = UpSampling2D(size=2, interpolation='nearest')(P3)
    P2 = add([C2, P3_up])
    return [P2, P3, P4, P5]


def cls_head(feats, n_filters, n_classes, n_anchors):
    cls_outputs = []
    for x in feats:
        for i in range(4):
            x = Conv_BN(x, n_filters, 3, strides=1, activation='relu')
        # head
        x = Conv2D(n_classes*n_anchors, 3, strides=1, padding='same')(x)
        cls_outputs.append(x)
    return cls_outputs


def box_head(feats, n_filters, n_anchors):
    box_outputs = []
    for x in feats:
        for i in range(4):
            x = Conv_BN(x, n_filters, 3, strides=1, activation='relu')
        # head
        x = Conv2D(4*n_anchors, 3, strides=1, padding='same')(x)
        box_outputs.append(x)
    return box_outputs


if __name__ == '__main__':

    model = retinaNet(input_shape=(224,224,3))
    model.summary()




