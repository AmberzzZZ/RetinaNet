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
    feature map: 8xP3-128xP7, backbone输出的特征图8xC3-32xC5，多出来的64xP6和128xP7通过3x3 stride2 conv得到

    本工程的head放在后面的cls和reg分支里面，
    本工程的conv后面加上了BN，retina源代码里面有，第三方keras实现里面没有
    本工程的feature map用的激活函数后的，原fpn论文也是，第三方keras实现用的激活函数前的


## cls
    focal loss, normailization
    anchor state {-1:ignore, 0:negative, 1:positive}，negative和positive的anchor参与运算

## box reg
    smooth l1 loss (huber loss)，前景positive参与运算