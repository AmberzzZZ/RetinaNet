## 官方tf源代码
    https://github.com/facebookresearch/Detectron/tree/master/detectron/modeling
## 第三方keras代码
    https://github.com/fizyr/keras-retinanet

## backbone
    resnet, resnext

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

## head
    cls head: [B,H,W,a,c]
    box head: [B,H,W,a,4], coord order: ofset[tx, ty, tw, th]
    每个level，每个grid，dense predict 9个anchor，one-hot logits & offset box
    shared across scales
    individual between cls & box

## rpn offset [t] ---- abs [b]
    tx = (bx - xa) / wa
    ty = (by - ya) / ha
    tw = log(bw / wa)
    th = log(bh / ha)
    可以看成是针对anchor的平移和缩放参数

## cls
    focal loss, normailization
    anchor state {-1:ignore, 0:negative, 1:positive}，negative和positive的anchor参与运算

## box reg
    smooth l1 loss (huber loss)，前景positive参与运算

## ytrue:
    cls: logits
    box: rela-origin-[xc,yc,w,h]
    match: 每个尺度上，每个gt box最多match一个anchor，
           每个anchor也最多match一个gt box（iou最大的那个）






