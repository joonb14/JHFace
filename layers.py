import tensorflow as tf
import math
from tensorflow.keras import backend as K

# class BatchNormalization(tf.keras.layers.BatchNormalization):
#     """Make trainable=False freeze BN for real (the og version is sad).
#        ref: https://github.com/zzh8829/yolov3-tf2
#     """
#     def call(self, x, training=False):
#         if training is None:
#             training = tf.constant(False)
#         training = tf.logical_and(training, self.trainable)
#         return super().call(x, training)

# https://github.com/peteryuX/arcface-tf2
class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'margin': self.margin,
            'logist_scale': self.logist_scale
        })
        return config
    
    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(tf.math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(tf.math.sin(self.margin), name='sin_m')
        self.th = tf.identity(tf.math.cos(tf.constant(math.pi) - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')
        
        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')
        ### modified == to tf.equal
        logists = tf.where(tf.math.equal(mask,1.), cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists

class AddMarginPenaltyLogists(tf.keras.layers.Layer):
    """AddMarginPenaltyLogists"""
    """need to change margin and logist_scale"""
    def __init__(self, num_classes, margin=0.35, logist_scale=64, **kwargs):
        super(AddMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'margin': self.margin,
            'logist_scale': self.logist_scale
        })
        return config
    
    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(tf.math.cos(self.margin), name='cos_m')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')

        cos_t_m = tf.subtract( cos_t , self.margin, name='cos_t_m' )

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')
        ### modified == to tf.equal
        logists = tf.where(tf.math.equal(mask,1.), cos_t_m, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'cosface_logist')

        return logists
    
    
# https://github.com/4uiiurz1/keras-arcface
class MulMarginPenaltyLogists(tf.keras.layers.Layer):
    """MulMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=1.35, logist_scale=30.0, **kwargs):
        super(MulMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.s = logist_scale
        self.m = margin
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'logist_scale': self.s,
            'margin': self.m
        })
        return config
    
    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])

    def call(self, embds, labels):
        x = embds
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.w, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.math.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.math.cos(self.m * theta)
        y = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        updated_logits = logits*self.s
        
        return updated_logits

