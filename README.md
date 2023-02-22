# JHFace

This project is based on the well implemented [arcface-tf2](https://github.com/peteryuX/arcface-tf2).
The things that rise error when converting the model to tflite was changed.

### Training the Model
For preparing data, follow the instructions provided in  [data-preparing](https://github.com/peteryuX/arcface-tf2#data-preparing)
Then checkout the [TensorFlow ArcFace.ipynb](https://github.com/joonb14/JHFace/blob/main/TensorFlow%20ArcFace.ipynb).

##### Backbones w/ ImageNet pretrained weights:

NasNet case, because of this [issue](https://github.com/keras-team/keras-applications/issues/78), we manually download the weight file and explicitly load it in [models.py](https://github.com/joonb14/JHFace/blob/main/models.py) file. We tried to provide pretrained weights for the MobileNet and EfficientNet models. However the official Keras implementations of the [MobileNetV3 has built in preprocessing layers inside the model](https://github.com/tensorflow/tensorflow/pull/47808#pullrequestreview-612848161). Also this accounts to [EfficientNet as well](https://github.com/tensorflow/tensorflow/pull/48276). So for the EfficientNet and EfficientNetLite, we used [Weights Transfer.ipynb](https://github.com/joonb14/JHFace/blob/main/Weights%20Transfer.ipynb) for extracting pretrained weights. To use the pretrained weights for NasNet, EfficientNet, EfficientNetLite **please download the weights using this [link](https://drive.google.com/file/d/1EriCfISIfeRAOso1DRRuQD8TFd_as35n/view?usp=sharing).** Then unzip it inside the path/to/JHFace/weights/ directory. All of the pretrained weights are provided by(or extracted from) [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)

* MobileNet
* MobileNetV2
* InceptionResNetV2
* InceptionV3
* ResNet50
* ResNet50V2
* ResNet101V2
* NASNetLarge
* NASNetMobile
* Xception
* MobileNetV3Large
* MobileNetV3Small
* EfficientNetLite0 ~ Lite6
* EfficientNetB0 ~ B7

##### Backbones w/o ImageNet pretrained weights:


We implemented MnasNet models looking at the [official code](https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py). If there's a bug, please tell us through the github issue page!

* MnasNetA1
* MnasNetB1
* MnasNetSmall

##### Loss Function

* [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)
* [CosFace](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html)
* [SphereFace](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf)
##### Configuration
in the [TensorFlow ArcFace.ipynb](https://github.com/joonb14/JHFace/blob/main/TensorFlow%20ArcFace.ipynb), we provided simple configuration values. To change the model backbone, just change the backbone_type parameter. To change the loss function, just change the head_type parameter.
```python
### MS1M dataset

batch_size = 128 # Initially 128
input_size = 112
embd_shape = 512
head_type = 'ArcHead' # 'ArcHead', 'CosHead', 'SphereHead'
# Backbones w/ pretrained weights:
#     MobileNet, MobileNetV2, InceptionResNetV2, InceptionV3, ResNet50, ResNet50V2, ResNet101V2, NASNetLarge, NASNetMobile, Xception, MobileNetV3Large, MobileNetV3Small, EfficientNetLite0~6, EfficientNetB0~7
# Backbones w/o pretrained weights:
#      MnasNetA1, MnasNetB1, MnasNetSmall 
backbone_type = 'EfficientNetLite0' 
w_decay=5e-4
num_classes = 85742 
dataset_len = 5822653 
if head_type == 'SphereHead':
    base_lr = 0.01 
    margin = 1.35
    logist_scale = 30.0 
elif head_type == 'CosHead':
    base_lr = 0.01 
    margin=0.35
    logist_scale=64
elif head_type == 'ArcHead':
    base_lr = 0.01 
    margin=0.5
    logist_scale=64
else:
    base_lr = 0.01 # initially 0.01
epochs = 20
save_steps = 1000
train_size = int(0.8 * dataset_len)
print(train_size)
steps_per_epoch = train_size // batch_size
print(steps_per_epoch)
val_size = dataset_len - train_size
print(val_size)
validation_steps = val_size // batch_size
print(validation_steps)
steps = 1
is_ccrop=False
binary_img=True
is_Adam = False
```

## Converting the Model to TensorFlow Lite

checkout the [TFLite conversion.ipynb](https://github.com/joonb14/arcface-tflite/blob/main/TFLite%20conversion.ipynb).
Int8 quantization is supported, we checked with MobileNetV2 and EfficientNet-lite0.

## For Face Verification
We downloaded the testing dataset from [here.](https://github.com/peteryuX/arcface-tf2#testing-dataset)
With the data use the [verification.ipynb](https://github.com/joonb14/JHFace/blob/main/verification.ipynb) for verification test.

## For Face Identificaiton

checkout the [Face Identification with Centroid Vector.ipynb](https://github.com/joonb14/JHFace/blob/main/Face%20Identification%20with%20Centroid%20Vector.ipynb).
