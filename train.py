from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from retinaNet import retinaNet
from dataGenerator import data_generator, anchors, get_anchors
import os

os.environ["CUDA_VISABLE_DEVICES"] = '-1'

if __name__ == '__main__':

    weight_dir = "weights/"
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    # data
    data_dir = "data/tmp"
    batch_size = 1
    input_shape = (512, 512)     # hw
    num_classes = 2
    strides = [8,16,32,64,128]
    train_generator = data_generator(data_dir, batch_size, input_shape, num_classes, anchors, strides=strides)
    # val_generator = data_generator(data_dir, batch_size, input_shape, num_classes, anchors, strides=strides)

    # model
    anchors = get_anchors(anchors, strides)
    n_anchors = anchors[0].shape[0]
    model = retinaNet(input_shape=input_shape+(1,), backbone='resnet', depth=50, fpn_filters=256,
                      n_classes=num_classes, n_anchors=n_anchors, anchors=anchors, strides=strides)
    lr = 1e-4
    decay = 5e-5
    model.compile(Adam(lr, decay), loss=lambda y_true, y_pred: y_pred, metrics=None)
    # for l in model.layers[:2]:
    #     l.trainable = False

    # train
    checkpoint = ModelCheckpoint(weight_dir+'/ep{epoch:03d}_loss{loss:.3f}.h5', monitor='loss',
                      save_best_only=True, save_weights_only=False, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=1)

    num_train = len(os.listdir(data_dir)) // 2
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=1,
                        epochs=30,
                        # validation_data=val_generator,
                        # validation_steps=1
                        callbacks=[checkpoint, reduce_lr, early_stopping])





