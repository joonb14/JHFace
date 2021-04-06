import collections
import re
from tensorflow.keras import layers, backend, Model
import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth',
    'stem_size', 'use_keras'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

# TODO(hongkuny): Consider rewrite an argument class with encoding/decoding.
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def _decode_block_string(block_string):
    """Gets a MNasNet block through a string notation of arguments.
    E.g. r2_k3_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
    k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
    o - output filters, se - squeeze/excitation ratio
    Args:
      block_string: a string, a string representation of block arguments.
    Returns:
      A BlockArgs instance.
    Raises:
      ValueError: if the strides option is not correctly specified.
    """
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value

    if 's' not in options or len(options['s']) != 2:
        raise ValueError('Strides options should be a pair of integers.')

    return BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]), int(options['s'][1])])

def decode(string_list):
    """Decodes a list of string notations to specify blocks inside the network.
    Args:
      string_list: a list of strings, each string is a notation of MnasNet
        block.
    Returns:
      A list of namedtuples to represent MnasNet blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
        blocks_args.append(_decode_block_string(block_string))
    return blocks_args

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_multiplier
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters

def _expand_conv(inputs, filters, data_format, name="expand_conv"):
    x = layers.Conv2D(
        filters, # channel 수를 expansion배 늘려준다
        1, # [1,1] filter
        padding='same',
        use_bias=False,
        data_format=data_format,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        activation=None,
        name="expand_conv"+name,
    )(inputs)
    return x

def _bn0(inputs, channel_axis, training, batch_norm_momentum, batch_norm_epsilon, name="bn0"):
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True, 
        name="bn0"+name,
        trainable=training)(inputs)
    return x

def _depth_wise_conv(inputs, kernel_size, strides, data_format, name="_depth_wise_conv"):
    x = layers.DepthwiseConv2D(
        kernel_size, # [kernel_size,kernel_size] filter
        strides=strides,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        padding='same',
        data_format=data_format,
        name="depth_wise_conv"+name,
        use_bias=False)(inputs)
    return x

def _bn1(inputs, channel_axis, training, batch_norm_momentum, batch_norm_epsilon, name="bn1"):
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True, 
        name="bn1"+name,
        trainable=training)(inputs)
    return x

def _se_reduce(inputs, input_filters, se_ratio, data_format, name="se_reduce"):
    num_reduced_filters = max(1, int(input_filters * se_ratio))
    x = layers.Conv2D(
        num_reduced_filters, # channel 수를 expansion배 늘려준다
        1, # [1,1] filter
        padding='same',
        data_format=data_format,
        use_bias=True,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="se_reduce"+name,
        activation=None
    )(inputs)
    return x

def _se_expand(inputs, filters, data_format, name="se_expand"):
    x = layers.Conv2D(
        filters, # channel 수를 expansion배 늘려준다
        1, # [1,1] filter
        padding='same',
        use_bias=True,
        data_format=data_format,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="se_expand"+name,
        activation=None
    )(inputs)
    return x

def _project_conv(inputs, filters, data_format, name="project_conv"):
    x = layers.Conv2D(
        filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding='same',
        use_bias=False,
        name="project_conv"+name,
        data_format=data_format)(inputs)
    return x

def _bn2(inputs, channel_axis, training, batch_norm_momentum, batch_norm_epsilon, name="bn2"):
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True, 
        trainable=training,
        name="bn2"+name,
    )(inputs)
    return x

def MnasBlock(block_args, global_params, block_id, training, name="MnasBlock"):
    def mnas_block(x_in):
        inputs = layers.Input(x_in.shape[1:])
        batch_norm_momentum = global_params.batch_norm_momentum
        batch_norm_epsilon = global_params.batch_norm_epsilon
        use_keras = global_params.use_keras
        data_format = global_params.data_format
        input_filters = block_args.input_filters
        filters = block_args.input_filters * block_args.expand_ratio
        output_filters = block_args.output_filters
        se_ratio = block_args.se_ratio
        kernel_size = block_args.kernel_size
        strides = block_args.strides

        if data_format == 'channels_first':
            channel_axis = 1
            spatial_dims = [2, 3]
        else:
            channel_axis = -1
            spatial_dims = [1, 2]
        has_se = (block_args.se_ratio is not None) and (
            block_args.se_ratio > 0) and (block_args.se_ratio <= 1)

    #         in_channels = backend.int_shape(inputs)[channel_axis]

        if block_args.expand_ratio != 1:
            x = _expand_conv(inputs, filters, data_format, name="_"+str(block_id))
            x = _bn0(x, channel_axis, training=training, batch_norm_momentum=batch_norm_momentum, batch_norm_epsilon=batch_norm_epsilon, name="_"+str(block_id))
            x = tf.nn.relu(x)
        else:
            x = inputs

        x = _depth_wise_conv(x, kernel_size, strides, data_format, name="_"+str(block_id))
        x = _bn1(x, channel_axis, training=training, batch_norm_momentum=batch_norm_momentum, batch_norm_epsilon=batch_norm_epsilon, name="_"+str(block_id))
        x = tf.nn.relu(x)

        if se_ratio:
            input_tensor = x
            se_tensor = tf.reduce_mean(x, spatial_dims, keepdims=True)
            se_tensor = _se_reduce(se_tensor, input_filters, se_ratio, data_format, name="_"+str(block_id))
            se_tensor = tf.nn.relu(se_tensor)
            se_tensor = _se_expand(se_tensor , filters, data_format, name="_"+str(block_id))
            x = tf.sigmoid(se_tensor) * input_tensor

        x = _project_conv(x, output_filters, data_format, name="_"+str(block_id))
        x = _bn2(x, channel_axis, training=training, batch_norm_momentum=batch_norm_momentum, batch_norm_epsilon=batch_norm_epsilon, name="_"+str(block_id))
        if block_args.id_skip:
            if all(s == 1 for s in strides) and input_filters == output_filters:
                x = tf.add(x, inputs)
        x = tf.identity(x)
        return Model(inputs,x, name=name+str(block_id))(x_in)
    return mnas_block

def mnasnet_b1(depth_multiplier=None):
    """Creates a mnasnet-b1 model.
    Args:
        depth_multiplier: multiplier to number of filters per layer.
    Returns:
        blocks_args: a list of BlocksArgs for internal MnasNet blocks.
        global_params: GlobalParams, global parameters for the model.
    """
    blocks_args = [
      'r1_k3_s11_e1_i32_o16_noskip', 'r3_k3_s22_e3_i16_o24',
      'r3_k5_s22_e3_i24_o40', 'r3_k5_s22_e6_i40_o80', 'r2_k3_s11_e6_i80_o96',
      'r4_k5_s22_e6_i96_o192', 'r1_k3_s11_e6_i192_o320_noskip'
    ]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.2,
        data_format='channels_last',
        num_classes=1000,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        stem_size=32,
        use_keras=True)
    return decode(blocks_args), global_params


def mnasnet_a1(depth_multiplier=None):
    """Creates a mnasnet-a1 model.
    Args:
        depth_multiplier: multiplier to number of filters per layer.
    Returns:
        blocks_args: a list of BlocksArgs for internal MnasNet blocks.
        global_params: GlobalParams, global parameters for the model.
    """
    blocks_args = [
      'r1_k3_s11_e1_i32_o16_noskip', 'r2_k3_s22_e6_i16_o24',
      'r3_k5_s22_e3_i24_o40_se0.25', 'r4_k3_s22_e6_i40_o80',
      'r2_k3_s11_e6_i80_o112_se0.25', 'r3_k5_s22_e6_i112_o160_se0.25',
      'r1_k3_s11_e6_i160_o320'
    ]
    global_params = GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.2,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None,
      stem_size=32,
      use_keras=True)
    
    return decode(blocks_args), global_params


def mnasnet_small(depth_multiplier=None):
    """Creates a mnasnet-a1 model.
    Args:
        depth_multiplier: multiplier to number of filters per layer.
    Returns:
        blocks_args: a list of BlocksArgs for internal MnasNet blocks.
        global_params: GlobalParams, global parameters for the model.
    """
    blocks_args = [
      'r1_k3_s11_e1_i16_o8', 'r1_k3_s22_e3_i8_o16',
      'r2_k3_s22_e6_i16_o16', 'r4_k5_s22_e6_i16_o32_se0.25',
      'r3_k3_s11_e6_i32_o32_se0.25', 'r3_k5_s22_e6_i32_o88_se0.25',
      'r1_k3_s11_e6_i88_o144'
    ]
    global_params = GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None,
      stem_size=8,
      use_keras=True)
  
    return decode(blocks_args), global_params


# def mnasnet_d1(depth_multiplier=None):
#     """Creates a jointly searched mnasnet backbone for mnas-fpn.
#     Args:
#         depth_multiplier: multiplier to number of filters per layer.
#     Returns:
#         blocks_args: a list of BlocksArgs for internal MnasNet blocks.
#         global_params: GlobalParams, global parameters for the model.
#     """
#     blocks_args = [
#       'r1_k3_s11_e9_i32_o24', 'r3_k3_s22_e9_i24_o36',
#       'r5_k3_s22_e9_i36_o48', 'r4_k5_s22_e9_i48_o96',
#       'r5_k7_s11_e3_i96_o96', 'r3_k3_s22_e9_i96_o80',
#       'r1_k7_s11_e6_i80_o320_noskip'
#     ]
#     global_params = GlobalParams(
#       batch_norm_momentum=0.99,
#       batch_norm_epsilon=1e-3,
#       dropout_rate=0.2,
#       data_format='channels_last',
#       num_classes=1000,
#       depth_multiplier=depth_multiplier,
#       depth_divisor=8,
#       min_depth=None,
#       stem_size=32,
#       use_keras=False)
  
#     return decode(blocks_args), global_params


# def mnasnet_d1_320(depth_multiplier=None):
#     """Creates a jointly searched mnasnet backbone for 320x320 input size.
#     Args:
#         depth_multiplier: multiplier to number of filters per layer.
#     Returns:
#         blocks_args: a list of BlocksArgs for internal MnasNet blocks.
#         global_params: GlobalParams, global parameters for the model.
#     """
#     blocks_args = [
#       'r3_k5_s11_e6_i32_o24', 'r4_k7_s22_e9_i24_o36',
#       'r5_k5_s22_e9_i36_o48', 'r5_k7_s22_e6_i48_o96',
#       'r5_k3_s11_e9_i96_o144', 'r5_k5_s22_e6_i144_o160',
#       'r1_k7_s11_e9_i160_o320'
#     ]

#     global_params = GlobalParams(
#       batch_norm_momentum=0.99,
#       batch_norm_epsilon=1e-3,
#       dropout_rate=0.2,
#       data_format='channels_last',
#       num_classes=1000,
#       depth_multiplier=depth_multiplier,
#       depth_divisor=8,
#       min_depth=None,
#       stem_size=32,
#       use_keras=False)
#     return decode(blocks_args), global_params

def MnasNetModel(input_shape=(224,224,3), include_top=False, weights=None, name="MnasNetA1"):
    if name=="MnasNetA1":
        _blocks_args, _global_params = mnasnet_a1()
    elif name=="MnasNetB1":
        _blocks_args, _global_params = mnasnet_b1()
    elif name=="MnasNetSmall":
        _blocks_args, _global_params = mnasnet_small()
    
    training=True 
    
    input_layer = layers.Input(shape=input_shape)
    
    outputs = layers.Conv2D(
        round_filters(_global_params.stem_size, _global_params), # channel 수를 expansion배 늘려준다
        3, # [3,3] filter
        2, # [2,2] stride
        padding='same',
        use_bias=False,
        data_format=_global_params.data_format,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        activation=None,
        name="Conv",
    )(input_layer)
    
    if _global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    
    outputs = layers.BatchNormalization(
        axis=channel_axis,
        momentum=_global_params.batch_norm_momentum,
        epsilon=_global_params.batch_norm_epsilon,
        fused=True, 
        trainable=training,
        name="bn0",
    )(outputs)
    
    outputs = tf.nn.relu(outputs)
    
    for idx, block_args in enumerate(_blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters, _global_params),
          output_filters=round_filters(block_args.output_filters, _global_params))

        # The first block needs to take care of stride and filter size increase.
        outputs = MnasBlock(block_args, _global_params, block_id=str(idx)+"_0", training=True)(outputs)

        if block_args.num_repeat > 1:
        # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
        for i in range(block_args.num_repeat - 1):
            outputs = MnasBlock(block_args, _global_params, block_id=str(idx)+"_"+str(i+1), training=True)(outputs)
    return Model(input_layer, outputs, name=name)
