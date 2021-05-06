import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pathlib import Path
import re
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from models import ArcFaceModel
from losses import softmax_loss
import dataset
import tensorflow as tf
import os
import logging
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from layers import ArcMarginPenaltyLogists
from tqdm import tqdm
from utils import l2_norm
import logging

tf.get_logger().setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"]="1"

### MS1M dataset

batch_size = 128 # Initially 128
input_size = 112
embd_shape = 512
head_type = 'ArcHead' # ''ArcHead', CosHead', 'SphereHead'
# Backbones w/ pretrained weights:
#     MobileNet, MobileNetV2, InceptionResNetV2, InceptionV3, ResNet50, ResNet50V2, ResNet101V2, NASNetLarge, NASNetMobile, Xception
#     But if you are trying to use NasNet, please check this issue first: https://github.com/keras-team/keras-applications/issues/78
#         We manually download the weight file and explicitly load it in models.py file
# Backbones w/o pretrained weights:
#     MobileNetV3Large, MobileNetV3Small, EfficientNetLite0~6, EfficientNetB0~7
backbone_type = 'MobileNetV2' 
w_decay=5e-4
num_classes = 85742 
dataset_len = 5822653 
base_lr = 0.01 # initially 0.01
epochs = 20
save_steps = 1000
train_size = int(0.8 * dataset_len)
print("train_size: ",train_size)
steps_per_epoch = train_size // batch_size
print("steps_per_epoch: ",steps_per_epoch)
val_size = dataset_len - train_size
print("val_size: ",val_size)
validation_steps = val_size // batch_size
print("validation_steps: ",validation_steps)
steps = 1
is_ccrop=False
binary_img=True
is_Adam = False
projection_head = False  # True
dgx = True

version = "Check"
    
if dgx:
    base_dir = "/raid/workspace/honghee/FaceRecognition/checkpoints/w_tfidentity/"
    if projection_head:
        save_name = f'ms1m_{backbone_type}_{head_type}_ProjectionHead_check/{version}/*'
    else:
        save_name = f'ms1m_{backbone_type}_{head_type}_check/{version}/*'
else:
    base_dir = "/hd/honghee/models/checkpoints/w_tfidentity/"
    save_name = f'ms1m_{backbone_type}_{head_type}_check/{version}/*'
    
# collect loss in checkpoints
file_list = []
for files in glob(base_dir+save_name):
    if not files.split('/')[-1].split('l_')[-1].split('.ckpt')[0] == 'checkpoint':
        loss = float( files.split('/')[-1].split('l_')[-1].split('.ckpt')[0] )
    file_list.append( loss  )
file_list.sort()

load_file_name = []
for files in glob(base_dir+save_name):
    if files.split('/')[-1].split('l_')[-1].split('.ckpt')[0] == 'checkpoint':
        pass
    elif file_list[0] == float( files.split('/')[-1].split('l_')[-1].split('.ckpt')[0] ) and files.split('/')[-1].split('l_')[-1].split('.ckpt')[-1]!='.index':
        load_file_name = files
best_checkpoint = load_file_name.split('.data')[0]
initial_epoch = int(load_file_name.split('e_')[-1].split('_')[0])-1
print("epoch: ",initial_epoch)
print("best checkpoint: ",best_checkpoint)


weight_file = best_checkpoint

model = ArcFaceModel(size=input_size,
                         backbone_type=backbone_type,
                         training=False)
model.load_weights(weight_file)
model.summary()

dataset_path = "/raid/workspace/jbpark/IJB-C/Gallery/"
id_list = os.listdir(dataset_path)
id_list.sort()
source_id = []
for id_name in id_list:
    source_id.append(int(id_list.index(id_name)))

subjects = id_list
label_int = source_id
embed_list = []
label_list = []
for subject in tqdm(subjects):
    template_paths = glob(dataset_path+subject+"/*")[0]
    img_paths = glob(template_paths+"/*")
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112,112))
        img = img.astype(np.float32) / 255.
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)

        embeds = l2_norm(model(img, training=False))
        embed_list.append(embeds[0].numpy())
        label_list.append(label_int[subjects.index(subject)])
embed_list = np.asarray(embed_list)
label_list = np.asarray(label_list)

save_path = "/raid/workspace/jbpark/IJB-C/numpy/"
if projection_head:
    save_name = f'ms1m_{backbone_type}_{head_type}_ProjectionHead/'
else:
    save_name = f'ms1m_{backbone_type}_{head_type}/'
Path(f'{save_path}{save_name}').mkdir(parents=True, exist_ok=True)
np.save(f'{save_path}{save_name}ijbc_gallery_vectors.npy', embed_list)
np.save(f'{save_path}{save_name}ijbc_gallery_labels.npy', label_list)

dataset_path = "/raid/workspace/jbpark/IJB-C/Probes/"

save_path = "/raid/workspace/jbpark/IJB-C/numpy/"
if projection_head:
    save_name = f'ms1m_{backbone_type}_{head_type}_ProjectionHead/'
else:
    save_name = f'ms1m_{backbone_type}_{head_type}/'
gallery_embed_list = np.load(f'{save_path}{save_name}ijbc_gallery_vectors.npy')
gallery_label_list = np.load(f'{save_path}{save_name}ijbc_gallery_labels.npy')

logger = logging.getLogger(__name__)

streamHandler = logging.StreamHandler()
if projection_head:
    fileHeandler = logging.FileHandler(f'./{backbone_type}_{head_type}_ProjectionHead.log')
else:
    fileHeandler = logging.FileHandler(f'./{backbone_type}_{head_type}.log')
logger.addHandler(streamHandler)
logger.addHandler(fileHeandler)
logger.setLevel(level=logging.DEBUG)

probe_id_list = os.listdir(dataset_path)
probe_id_list.sort()
source_id = []
for id_name in probe_id_list:
    source_id.append(int(probe_id_list.index(id_name)))

subjects = probe_id_list
label_int = source_id
for subject in tqdm(subjects):
    average_list = []
    template_paths = glob(dataset_path+subject+"/*")
    logger.info(f'============={subject}=============')
    for template_path in template_paths:
        template_name = template_path.split("/")[-1]
        img_paths = glob(template_path+"/*")
        true_count=0
        false_count=0
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112,112))
            img = img.astype(np.float32) / 255.
            if len(img.shape) == 3:
                img = np.expand_dims(img, 0)

            embeds = l2_norm(model(img, training=False))

            embed_list = []
            label_list = []
            for idx, gallery_emb in enumerate(gallery_embed_list):
                embed_list.append(np.dot(embeds,gallery_emb)/(np.linalg.norm(embeds)*np.linalg.norm(gallery_emb)))
            est = probe_id_list[gallery_label_list[np.argmax(embed_list)]]
            if(est == subject):
                true_count+=1
            else:
                false_count+=1
        average = true_count/(true_count+false_count)
        average_list.append(average)
        logger.info(f'template ID: {template_name}, result | {average}')
    logger.info(f'average of {subject}: {np.mean(average_list)}')

### Calculating Average

df = pd.read_csv("/raid/workspace/jihyun/IJBC/IJB/IJB-C/protocols/archive/ijbc_1N_probe_mixed.csv")

class_list = []
template_id_list = []
accuracy_list = []
image_num_list = []
class_name = None
if projection_head:
    file_name = f'{backbone_type}_{head_type}_ProjectionHead.log'
else:
    file_name = f'{backbone_type}_{head_type}.log'
with open(file_name, 'r') as f:
    for line in f:
        if '===' in line:
            class_name = line.split('=============')[1]
        elif 'template ID:' in line:
            template_id=re.split(r'[ ,:|\n]', line)
            template_id_list.append(int(template_id[3]))
            accuracy_list.append(float(template_id[8]))
            temp = df.loc[df['TEMPLATE_ID'] == int(template_id[3])]
            image_num_list.append(temp.shape[0])
            class_list.append(class_name)

result_df = pd.DataFrame(list(zip(class_list, template_id_list, accuracy_list, image_num_list)), columns = ['SUBJECT', 'TEMPLATE_ID', 'ACCURACY', 'IMAGE_NUM'])
if projection_head:
    file_name = f'{backbone_type}_{head_type}_ProjectionHead.csv'
else:
    file_name = f'{backbone_type}_{head_type}.csv'
result_df.to_csv(file_name,header=True,index=False)
avg_df = pd.DataFrame(result_df, columns= ['ACCURACY', 'IMAGE_NUM'])
sum = (avg_df['ACCURACY'] * avg_df['IMAGE_NUM']).sum()
acc = sum/avg_df['IMAGE_NUM'].sum()
print("Average Accuracy: ",acc)