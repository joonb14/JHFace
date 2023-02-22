import tensorflow as tf
import math
from tensorflow.keras import backend as K
import tensorflow_probability as tfp


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

class CurMarginPenaltyLogists(tf.keras.layers.Layer):
    """CurMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(CurMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
        self.t = tf.Variable(tf.zeros(1), trainable=False, name='t')
    
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
        self.t.assign(tf.math.reduce_mean(cos_t) * 0.01 + (1 - 0.01) * self.t)
        cos_t = tf.where(tf.math.greater(cos_t , cos_mt), cos_t * (self.t + cos_t), cos_t)
        
        onehot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')
        logists = tf.where(tf.math.equal(onehot,1.), cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'curface_logist')
        
        return logists

# https://github.com/samisoto/keras_cosine_based_loss
class AdaMarginPenaltyLogists(tf.keras.layers.Layer):
    """AdaMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(AdaMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
#         self.t = tf.Variable(tf.zeros(1), trainable=False, name='t')
    
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
#         self.cos_m = tf.identity(tf.math.cos(self.margin), name='cos_m')
#         self.sin_m = tf.identity(tf.math.sin(self.margin), name='sin_m')
#         self.th = tf.identity(tf.math.cos(tf.constant(math.pi) - self.margin), name='th')
#         self.mm = tf.multiply(self.sin_m, self.margin, name='mm')
        
        self.s = tf.Variable(tf.math.sqrt(2.) * tf.math.log(tf.cast(self.num_classes - 1, tf.float32)), trainable=False)
        self.correct_cos_mean = tf.Variable(0., trainable=False)
        
    def call(self, embds, labels):
           
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
    
        # labels.shape = (batch_size,1)
        mask = tf.one_hot(labels, depth=cos_t.shape[-1], name='one_hot_mask')
#         mask = tf.one_hot(labels, depth=self.num_classes, name='one_hot_mask')
        # mask.shape = (batch_size,1,num_classes)
#         mask = tf.squeeze(mask, axis=1)
        # mask.shape = (batch_size,num_classes)

        correct_cos_t = tf.reduce_sum(mask * cos_t, axis=1)
        self.correct_cos_mean.assign(tf.reduce_mean(correct_cos_t))
        
        Bavg = (tf.ones_like(mask) - mask) * tf.exp(self.s * cos_t)
        # summarize num_classes
        Bavg = tf.reduce_sum(Bavg, axis=1)
        # average batch
        Bavg = tf.reduce_mean(Bavg, axis=0, name='B_avg')

        cos_med = tfp.stats.percentile(correct_cos_t, q=50, interpolation='midpoint', name='correct_cos_median')
        self.s.assign(tf.math.log(Bavg) / tf.maximum(1 / tf.math.sqrt(2.), cos_med))

        self.add_metric(self.s, name="s")
        self.add_metric(self.correct_cos_mean, name='correct_cos_mean')

        logits = cos_t * self.s

        return logits
    
class CadMarginPenaltyLogists(tf.keras.layers.Layer):
    """CadMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(CadMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
        self.t = tf.Variable(tf.zeros(1), trainable=False, name='t')
    
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
        
        self.s = tf.Variable(tf.math.sqrt(2.) * tf.math.log(tf.cast(self.num_classes - 1, tf.float32)), trainable=False)
        self.correct_cos_mean = tf.Variable(0., trainable=False)
        
    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        
        # adaptive scaling
        mask = tf.one_hot(labels, depth=cos_t.shape[-1], name='one_hot_mask')
        correct_cos_t = tf.reduce_sum(mask * cos_t, axis=1)
        self.correct_cos_mean.assign(tf.reduce_mean(correct_cos_t))
        
        Bavg = (tf.ones_like(mask) - mask) * tf.exp(self.s * cos_t)
        # summarize num_classes
        Bavg = tf.reduce_sum(Bavg, axis=1)
        # average batch
        Bavg = tf.reduce_mean(Bavg, axis=0, name='B_avg')

        cos_med = tfp.stats.percentile(correct_cos_t, q=50, interpolation='midpoint', name='correct_cos_median')
        self.s.assign(tf.math.log(Bavg) / tf.maximum(1 / tf.math.sqrt(2.), cos_med))

        self.add_metric(self.s, name="s")
        self.add_metric(self.correct_cos_mean, name='correct_cos_mean')
        ## scaling end
        
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')
        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')
        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)
        self.t.assign(tf.math.reduce_mean(cos_t) * 0.01 + (1 - 0.01) * self.t)
        cos_t = tf.where(tf.math.greater(cos_t , cos_mt), cos_t * (self.t + cos_t), cos_t)
        
        onehot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')
        logists = tf.where(tf.math.equal(onehot,1.), cos_mt, cos_t)
#         logists = tf.multiply(logists, self.logist_scale, 'curface_logist')
        logists = tf.multiply(logists, self.s, 'cadface_logist')
        
        return logists
    
    




