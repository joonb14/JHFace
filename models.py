import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,

    BatchNormalization,  #Added
)
from tensorflow.keras.applications import (
    MobileNet,
    MobileNetV2,
    InceptionResNetV2,
    InceptionV3,
    ResNet50,
    ResNet50V2,
    ResNet101V2,
    NASNetLarge,
    NASNetMobile,
    Xception
)
from layers import (
    #BatchNormalization,
    ArcMarginPenaltyLogists,
    AddMarginPenaltyLogists,
    MulMarginPenaltyLogists,
    CurMarginPenaltyLogists,
    CadMarginPenaltyLogists,
    AdaMarginPenaltyLogists,
    SvxMarginPenaltyLogists
)
from backbone.efficientnet_lite  import (
    EfficientNetLite0,
    EfficientNetLite1,
    EfficientNetLite2,
    EfficientNetLite3,
    EfficientNetLite4,
    EfficientNetLite5,
    EfficientNetLite6
) 
from backbone.efficientnet  import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7
) 
from backbone.mobilenet_v3  import (
    MobileNetV3Small,
    MobileNetV3Large
)
from backbone.mnasnet import (
    MnasNetModel
)

WEIGHTS_DIR = "./weights/"

def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def Backbone(backbone_type='ResNet50V2', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'ResNet50V2':
            return ResNet50V2(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'ResNet101V2':
            return ResNet101V2(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'InceptionResNetV2':
            return InceptionResNetV2(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'InceptionV3':
            return InceptionV3(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNet':
            return MobileNet(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'NASNetLarge':
            model = NASNetLarge(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"nasnet_large_no_top.h5")
            return model(x_in)
        elif backbone_type == 'NASNetMobile':
            model = NASNetMobile(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"nasnet_mobile_no_top.h5")
            return model(x_in)
        elif backbone_type == 'Xception':
            return Xception(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'MobileNetV3Small':
            model = MobileNetV3Small(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"mobilenet_v3_small_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'MobileNetV3Large':
            model = MobileNetV3Large(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"mobilenet_v3_large_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetLite0':
            model = EfficientNetLite0(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnet_lite0_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetLite1':
            model = EfficientNetLite1(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnet_lite1_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetLite2':
            model = EfficientNetLite2(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnet_lite2_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetLite3':
            model = EfficientNetLite3(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnet_lite3_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetLite4':
            model = EfficientNetLite4(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnet_lite4_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetLite5':
            model = EfficientNetLite5(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnet_lite5_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetLite6':
            model = EfficientNetLite6(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnet_lite6_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetB0':
            model = EfficientNetB0(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnetb0_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetB1':
            model = EfficientNetB1(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnetb1_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetB2':
            model = EfficientNetB2(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnetb2_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetB3':
            model = EfficientNetB3(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnetb3_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetB4':
            model = EfficientNetB4(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnetb4_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetB5':
            model = EfficientNetB5(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnetb5_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetB6':
            model = EfficientNetB6(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnetb6_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'EfficientNetB7':
            model = EfficientNetB7(input_shape=x_in.shape[1:], include_top=False,
                               weights=None)
            model.load_weights(WEIGHTS_DIR+"efficientnetb7_notop.ckpt")
            return model(x_in)
        elif backbone_type == 'MnasNetA1':
            return MnasNetModel(input_shape=x_in.shape[1:], include_top=False,
                               weights=None, name="MnasNetA1")(x_in)
        elif backbone_type == 'MnasNetB1':
            return MnasNetModel(input_shape=x_in.shape[1:], include_top=False,
                               weights=None, name="MnasNetB1")(x_in)
        elif backbone_type == 'MnasNetSmall':
            return MnasNetModel(input_shape=x_in.shape[1:], include_top=False,
                               weights=None, name="MnasNetSmall")(x_in)
        else:
            raise TypeError('backbone_type error!')
    return backbone


def OutputLayer(embd_shape, w_decay=5e-4,trainable=False, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization(trainable=trainable, name='output_batch_norm_1')(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization(trainable=trainable, name='output_batch_norm_2')(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


def ArcHead(num_classes, margin=0.5, logist_scale=64, projection_head=False, name='ArcHead'): 
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
#         nonlinear projection head
        if projection_head:
            x = Dense(32, activation='relu')(x)
#             x = Dense(64, activation='relu', use_bias=True, bias_initializer='zeros')(x)
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head

def CosHead(num_classes, margin=0.35, logist_scale=64, name='CosHead'):
    """Cos Head"""
    def cos_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = AddMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return cos_head

def SphereHead(num_classes, margin=1.35, logist_scale=30, name='SphereHead'):
    """Sphere Head"""
    def sphere_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:], dtype=tf.int32)
        x = MulMarginPenaltyLogists(num_classes=num_classes, margin=margin, logist_scale=logist_scale)(x, y)
#         x = MulMarginPenaltyLogists_practice(num_classes=num_classes, margin=margin, logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return sphere_head

def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head

def CurHead(num_classes, margin=0.35, logist_scale=64, name='CurHead'):
    """Cur Head"""
    def cur_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:], dtype=tf.int32)
        x = CurMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return cur_head

def CadHead(num_classes, margin=0.35, logist_scale=64, name='CadHead'):
    """Cad Head"""
    def cad_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:], dtype=tf.int32)
        x = CadMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return cad_head

def AdaHead(num_classes, margin=0.35, logist_scale=64, name='AdaHead'):
    """Ada Head"""
    def ada_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:], dtype=tf.int32)
        x = AdaMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return ada_head

def SvxHead(num_classes, margin=0.35, logist_scale=64, t=0.2, name='SvxHead'):
    """Svx Head"""
    def svx_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:], dtype=tf.int32)
        x = SvxMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale, t=t)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return svx_head


def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False, projection_head=False):  
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    embds = OutputLayer(embd_shape, w_decay=w_decay, trainable=training)(x)
#     if projection_head:
#         embds  = Dense(128, activation='relu')(embds)
    if training:
        assert num_classes is not None
        labels = Input([], name='label', dtype=tf.int32)
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale, projection_head=projection_head)(embds, labels)
        elif head_type == 'CosHead':
            logist = CosHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        elif head_type == 'SphereHead':
            logist = SphereHead(num_classes=num_classes, margin=margin, 
                                logist_scale=logist_scale)(embds, labels)
        elif head_type == 'CurHead':
            logist = CurHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        elif head_type == 'CadHead':
            logist = CadHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        elif head_type == 'AdaHead':
            logist = AdaHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        elif head_type == 'SvxHead':
            logist = SvxHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        else:
            logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, embds, name=name)
