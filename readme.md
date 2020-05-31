## 官方源代码
    https://github.com/facebookresearch/Detectron/tree/master/detectron/modeling

## resnext
    分组卷积keras没有

## fpn
    fpn原论文实现是
    lateral connection: 1x1x256 conv
    top-down connection: UpSampling2D('nearest')
    head: 3x3x256 conv
    没引入BN
    本工程的head放在后面的cls和reg分支里面，加上了BN

## cls
    focal loss, normailization

## box reg
    l1 loss