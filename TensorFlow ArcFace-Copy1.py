#!/usr/bin/env python
# coding: utf-8

# # TensorFlow ArcFace

# ### Ref https://github.com/peteryuX/arcface-tf2

# In[1]:


from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from models import ArcFaceModel
# from losses import SoftmaxLoss
from losses import softmax_loss
import dataset
# import tensorflow as tf
import os
import logging
# from tensorflow.python import debug as tf_debug
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_v2_behavior()
# tf debugger V2
tf.config.set_soft_device_placement(True)
tf.debugging.experimental.enable_dump_debug_info(
    dump_root="logs/arcface/tfdbg2",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)

tf.get_logger().setLevel(logging.ERROR)
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"s

gpu_num = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


# In[2]:


get_ipython().system('echo $CUDA_VISIBLE_DEVICES')


# In[3]:


### IJB-C Dataset
# batch_size = 128
# input_size = 112
# embd_shape = 512
# head_type = 'ArcHead'
# backbone_type = 'MobileNetV2'
# w_decay=5e-4
# num_classes = 3584 
# base_lr = 0.01
# dataset_len = 13033 
# epochs = 100
# steps_per_epoch = dataset_len // batch_size

### MS1M dataset
num_classes = 85742 
dataset_len = 5822653
batch_size = 128 # Initially 128
input_size = 112
embd_shape = 512
train_size = int(0.8 * dataset_len)
steps_per_epoch = train_size // batch_size
val_size = dataset_len - train_size
validation_steps = val_size // batch_size

w_decay=5e-4
epochs = 100

save_steps = 1000
steps = 1
is_ccrop=False
binary_img=True

is_Adam = False   # True
projection_head = True  # False
dgx = True

head_type = 'ArcHead' # ''ArcHead', CosHead', 'SphereHead', 'CurHead', 'SvxHead'
# Backbones w/ pretrained weights:
#     MobileNet, MobileNetV2, InceptionResNetV2, InceptionV3, ResNet50, ResNet50V2, ResNet101V2, NASNetLarge, NASNetMobile, Xception, MobileNetV3Large, MobileNetV3Small, EfficientNetLite0~6, EfficientNetB0~7
# Backbones w/o pretrained weights:
#      MnasNetA1, MnasNetB1, MnasNetSmall 
backbone_type = 'ResNet50' 

if head_type == 'SphereHead':
    base_lr = 0.01 
    margin = 1.35
    logist_scale = 30.0 
elif head_type == 'CosHead':
    base_lr = 0.01 
    margin=0.35
    logist_scale=64
elif head_type == 'ArcHead':
#     base_lr = 0.001 
    base_lr = 0.1 
    margin=0.5
    logist_scale=64
elif head_type == 'CurHead':
    base_lr = 0.01 
    margin=0.5
    logist_scale=64
elif head_type == 'SvxHead':
    base_lr = 0.1 
    margin=0.5
    logist_scale=64
else:
    base_lr = 0.01 # initially 0.01
    
print(head_type)
print(backbone_type)
print("projection head:", projection_head)
print("Adam:", is_Adam)
print("epoch:", epochs)
print("batch size:", batch_size)
print("Learning Rate:", base_lr)


# In[4]:


# train_data_dir = "/raid/workspace/jbpark/IJB-C_Asian/"
# tfrecord_name = train_data_dir+'ijbc_bin.tfrecord'
if dgx:
    train_data_dir = "/raid/workspace/jbpark/ms1m/"
    tfrecord_name = f'{train_data_dir}ms1m_bin.tfrecord'
else:
    train_data_dir = "/hd/jbpark/dataset/ms1m/"
    tfrecord_name = f'{train_data_dir}ms1m_bin.tfrecord'


train_dataset, val_dataset = dataset.load_tfrecord_dataset(
    tfrecord_name, batch_size, train_size=train_size, binary_img=binary_img, 
    is_ccrop=is_ccrop)

# print("data: ", train_dataset)


strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{gpu_num}"])
# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

with strategy.scope():
    model = ArcFaceModel(size=input_size,
                             backbone_type=backbone_type,
                             num_classes=num_classes,
                             margin=margin, 
                             logist_scale=logist_scale,
                             head_type=head_type,
                             embd_shape=embd_shape,
                             w_decay=w_decay,
                             training=True) #projection_head = projection_head
    model.summary()

    learning_rate = tf.constant(base_lr)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                 decay_steps=300000, 
                                                   decay_rate=0.1,staircase=True)
    if is_Adam:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule)
    else:
#         optimizer = tf.keras.optimizers.SGD(
#             learning_rate=learning_rate, momentum=0.9)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9)

    loss_fn = softmax_loss

    model.compile(optimizer=optimizer, loss=loss_fn)


# In[ ]:


from pathlib import Path
if dgx:
    base_dir = "/raid/workspace/honghee/FaceRecognition/"
    if projection_head:
        save_name = f'ms1m_{backbone_type}_{head_type}_ProjectionHead_check/'
    else:
        save_name = f'ms1m_{backbone_type}_{head_type}_check/'
else:
    base_dir = "/hd/honghee/models/"
    save_name = f'ms1m_{backbone_type}_{head_type}_check/'

if is_Adam:
    version = "Adam"
else:
    version = "SGD"

Path(f'{base_dir}checkpoints/w_tfidentity/{save_name}{version}').mkdir(parents=True, exist_ok=True)

### MS1M dataset
tb_callback = TensorBoard(log_dir='logs/arcface/Proj_Head', histogram_freq=0,
                          write_graph=False, write_images=True,
                                  update_freq = 1000,
                                  profile_batch=0)
tb_callback._total_batches_seen = steps
tb_callback._samples_seen = steps * batch_size
check_dir = f'{base_dir}checkpoints/w_tfidentity/{save_name}{version}'
mc_callback = ModelCheckpoint(
            check_dir+'/e_{epoch}_l_{loss}.ckpt',
            save_freq = save_steps, verbose=1,
            save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# callbacks = [mc_callback, tb_callback]
callbacks = [mc_callback, tb_callback, early_stopping]
# callbacks = [mc_callback, tb_callback, early_stopping, lr_scheduler]

### IJB-C Dataset
# callbacks = [
#     ModelCheckpoint(
#         base_dir+"checkpoints/"+save_name+".ckpt", 
# #         monitor='val_accuracy', 
#         monitor='loss', 
#         verbose=1, 
#         save_best_only=True, 
#         save_weights_only = True,
#         mode='min'
#     ),
#     EarlyStopping(
# #         monitor='val_accuracy', 
#         monitor='loss', 
#         patience=15, 
#         min_delta=0.001, 
#         mode='min'
#     )
# ]

model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=epochs, 
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)


# ### Resume training with latest checkpoint

# In[4]:


from glob import glob
from pathlib import Path

if is_Adam:
    version = "Adam"
else:
    version = "SGD"

if dgx:
    base_dir = "/raid/workspace/honghee/FaceRecognition/checkpoints/w_tfidentity/"
    # save_name = "ms1m_mobilenet_check/SGD/*"
    if projection_head:
        save_name = f'ms1m_{backbone_type}_{head_type}_ProjectionHead_check/{version}/*'
    else:
        save_name = f'ms1m_{backbone_type}_{head_type}_check/{version}/*'
else:
    base_dir = "/hd/honghee/models/checkpoints/w_tfidentity/"
    save_name = f'ms1m_{backbone_type}_{head_type}_check/{version}/*'
    
## collect loss in checkpoints
file_list = []
for files in glob(f'{base_dir}{save_name}'):
    if not files.split('/')[-1].split('l_')[-1].split('.ckpt')[0] == 'checkpoint':
        loss = float( files.split('/')[-1].split('l_')[-1].split('.ckpt')[0] )
    file_list.append( loss  )
file_list.sort()

load_file_name = []
for files in glob(f'{base_dir}{save_name}'):
    if files.split('/')[-1].split('l_')[-1].split('.ckpt')[0] == 'checkpoint':
        pass
    elif file_list[0] == float( files.split('/')[-1].split('l_')[-1].split('.ckpt')[0] ) and files.split('/')[-1].split('l_')[-1].split('.ckpt')[-1]!='.index':
        load_file_name = files
best_checkpoint = load_file_name.split('.data')[0]
initial_epoch = int(load_file_name.split('e_')[-1].split('_')[0])-1
print(initial_epoch)
print(best_checkpoint)


# In[5]:


# train_data_dir = "/raid/workspace/jbpark/IJB-C_Asian/"
# tfrecord_name = train_data_dir+'ijbc_bin.tfrecord'

if dgx:
    train_data_dir = "/raid/workspace/jbpark/ms1m/"
    tfrecord_name = f'{train_data_dir}ms1m_bin.tfrecord'
else:
    train_data_dir = "/hd/jbpark/dataset/ms1m/"
    tfrecord_name = f'{train_data_dir}ms1m_bin.tfrecord'


train_dataset, val_dataset = dataset.load_tfrecord_dataset(
    tfrecord_name, batch_size, train_size=train_size, binary_img=binary_img,
    is_ccrop=is_ccrop)

if dgx:
    base_dir = "/raid/workspace/honghee/FaceRecognition/"
    if projection_head:
        save_name = f'ms1m_{backbone_type}_{head_type}_ProjectionHead_check/'
    else:
        save_name = f'ms1m_{backbone_type}_{head_type}_check/'
else:
    base_dir = "/hd/honghee/models/"
    save_name = f'ms1m_{backbone_type}_{head_type}_check/'

Path(f'{base_dir}checkpoints/w_tfidentity/{save_name}{version}').mkdir(parents=True, exist_ok=True)

# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:{gpu_num}"])
strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{gpu_num}"])

with strategy.scope():

    model = ArcFaceModel(size=input_size,
                             backbone_type=backbone_type,
                             num_classes=num_classes,
                             head_type=head_type,
                             embd_shape=embd_shape,
                             w_decay=w_decay,
                             training=True,
                             projection_head=projection_head)
    model.load_weights(best_checkpoint)

    learning_rate = tf.constant(base_lr)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                 decay_steps=300000, 
                                                   decay_rate=0.1,staircase=True)
    if is_Adam:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule)
    else:
#         optimizer = tf.keras.optimizers.SGD(
#             learning_rate=learning_rate, momentum=0.9)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9)

    loss_fn = softmax_loss

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.summary()
    
tb_callback = TensorBoard(log_dir='logs/arcface/',
                                  update_freq = batch_size * 5,
                                  profile_batch=0)
tb_callback._total_batches_seen = steps
tb_callback._samples_seen = steps * batch_size
check_dir = f'{base_dir}checkpoints/w_tfidentity/{save_name}{version}'
mc_callback = ModelCheckpoint(
            check_dir+'/e_{epoch}_l_{loss}.ckpt',
            save_freq = save_steps, verbose=1,
            save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# callbacks = [mc_callback, tb_callback]
callbacks = [mc_callback, tb_callback, early_stopping]


# In[ ]:


model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=epochs, 
    initial_epoch=initial_epoch,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)


# In[ ]:




