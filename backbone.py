# resnet & resnext, 50 & 101
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, concatenate
from keras.models import Model
import tensorflow as tf


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
n_filters = [256, 512, 1024, 2048]


def resnet(input_tensor=None, input_shape=(224,224,3), depth=50):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='relu')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    features = []
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if i!=0 and j==0 else 1
            x = res_block(x, n_filters[i], strides)
        features.append(x)

    # model
    model = Model(inpt, features)

    return model


def resnext(input_tensor=None, input_shape=(224,224,3), depth=50, C=32):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='relu')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    features = []
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if i!=0 and j==0 else 1
            x = resnext_block(x, n_filters[i], strides, C)
        features.append(x)

    # model
    model = Model(inpt, features)

    return model


def resnext_block(x, n_filters, strides, C=32):
    inpt = x
    # group residual
    x = Conv_BN(inpt, n_filters//2, 1, strides=strides, activation='relu')
    x = group_conv(x, C, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def group_conv(x, n_groups, kernel_size, strides, activation='relu'):
    # conv-bn-act by group
    n_filters = x._keras_shape[-1] // n_groups
    group_x = tf.split(x, n_groups, axis=-1)
    group_feature = []
    for g in group_x:
        x = Conv_BN(x, n_filters, 3, strides, activation)
        group_feature.append(x)
    x = concatenate(group_feature, axis=-1)
    return x


def res_block(x, n_filters, strides):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//4, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//4, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = resnext(input_shape=(224,224,3), depth=50)
    model.summary()

