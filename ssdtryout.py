import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #stavi 3
import tensorflow as tf

main_path = './images/'

artdict = {}
num = 0
for element in os.listdir(main_path):
    element_path=os.path.join(main_path, element)
    num = 0
    for elementjpg in os.listdir(element_path):
        element_pathjpg=os.path.join(main_path, elementjpg)
        if element_pathjpg.endswith("jpg"):
            num += 1
    artdict.update({element: num})

sortedart = sorted(artdict.items(), key=lambda x:x[1], reverse=True)
sorteddict = dict(sortedart)
print(sorteddict);

classw = {}
numpaintings = sum(sorteddict.values())
print(numpaintings)

i = 0
for x, y in sorteddict.items():
    classweight = numpaintings/(10*y)
    classw.update({i: classweight})
    i = i + 1
    
print(classw)

img_size = (128, 128)
batch_size = 64

from keras.utils import image_dataset_from_directory

Xtrain = image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

classes = Xtrain.class_names
print(classes)

#%%
N = 10
plt.figure(figsize=(20,15))
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')
        
#%%
from keras import layers
from keras import Sequential
data_augmentation = Sequential(
    [
     layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
     layers.RandomRotation(0.25),
     layers.RandomZoom(0.1),
     ]
)

N = 10
plt.figure(figsize=(15,10))
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')


#%%
from keras import Sequential
from keras import layers
from keras import Model

from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, MaxPooling2D, Flatten, Dense, \
                                       AveragePooling2D, RandomFlip,\
                                    RandomRotation, RandomZoom, Dropout, Rescaling

from tensorflow.keras import regularizers
# from keras.callbacks import ModelCheckpoint


num_classes = len(classes)

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(128, 128, 3))
    num_filters = 32
    t = RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3))(inputs)
    t = RandomRotation(0.25)(t)
    t = RandomZoom(0.15)(t)
    t = Rescaling(1./255)(t)
    t = BatchNormalization()(t)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 2, 2, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D()(t)
    t = Conv2D(512, 5)(t)
    t = relu_bn(t)
    t = MaxPooling2D()(t)
    # t = Conv2D(512, 1)(t)
    # t = relu_bn(t)
    t = Flatten()(t)
    t = Dense(256, activation='relu', kernel_regularizer=regularizers.L1(l1=5*1e-3))(t)
    t = Dropout(0.1)(t)
    t = Dense(128, activation='relu')(t)
    outputs = Dense(10, activation='softmax')(t)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_res_net()
model.summary()

# checkpoint_filepath = './checkpoint'
# checkpoint = ModelCheckpoint(checkpoint_filepath)

#%%

history = model.fit(Xtrain,
                    batch_size=batch_size,
                    epochs=150,
                    validation_data=Xval,
                    class_weight=classw,
                    verbose=1)

#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

