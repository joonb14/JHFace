import tensorflow as tf
import math


# class BatchNormalization(tf.keras.layers.BatchNormalization):
#     """Make trainable=False freeze BN for real (the og version is sad).
#        ref: https://github.com/zzh8829/yolov3-tf2
#     """
#     def call(self, x, training=False):
#         if training is None:
#             training = tf.constant(False)
#         training = tf.logical_and(training, self.trainable)
#         return super().call(x, training)


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
    
    ## need to modify
    
    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(tf.math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(tf.math.sin(self.margin), name='sin_m')
        self.th = tf.identity(tf.math.cos(tf.constant(math.pi) - self.margin), name='th')
#         self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_t_m = tf.subtract( cos_t , self.margin, name='cos_t_m' )

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')
        ### modified == to tf.equal
        logists = tf.where(tf.math.equal(mask,1.), cos_t_m, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'cosface_logist')

        return logists
    
class MulMarginPenaltyLogists(tf.keras.layers.Layer):
    """MulMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=4, **kwargs):
        super(MulMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'margin': self.margin
        })
        return config
    
    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(tf.math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(tf.math.sin(self.margin), name='sin_m')
        self.th = tf.identity(tf.math.cos(tf.constant(math.pi) - self.margin), name='th')
#         self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        l = 0.
#         normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')
        embeddings_norm = tf.norm(embds, axis=1)

        orginal_logits = tf.matmul(embds, normed_w, name='orginal_logits')
        
        N = embds.get_shape()[0] # get batch_size
        single_sample_label_index = tf.stack([tf.constant(list(range(N)), tf.int64), labels], axis=1)
        selected_logits = tf.gather_nd(orginal_logits, single_sample_label_index) 
        cos_theta = tf.div(selected_logits, embeddings_norm)
        cos_theta_power = tf.square(cos_theta)
        cos_theta_biq = tf.pow(cos_theta, 4)
        sign0 = tf.sign(cos_theta)
        sign3 = tf.multiply(tf.sign(2*cos_theta_power-1), sign0)
        sign4 = 2*sign0 + sign3 -3
        result=sign3*(8*cos_theta_biq-8*cos_theta_power+1) + sign4
        
        margin_logits = tf.multiply(result, embeddings_norm)
        f = 1.0/(1.0+l)
        ff = 1.0 - f
        combined_logits = tf.add(orginal_logits, tf.scatter_nd(single_sample_label_index, tf.subtract(margin_logits, selected_logits), orginal_logits.get_shape()))
        
        updated_logits = ff*orginal_logits + f*combined_logits
        
        return updated_logits
