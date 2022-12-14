from __future__ import print_function,division
from builtins import range,input

import tensorflow
from keras.layers import Input,Dense,Lambda,Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

#resize data
IMAGE_SIZE = [100,100]

#training config
epochs = 5
batch_size = 32

train_path = 'C:/Users/Emmanuel Alo/Documents/vgg data/fruits-360-original-size/fruits-360-original-size\Training'
test_path = 'C:/Users/Emmanuel Alo/Documents/vgg data/fruits-360-original-size/fruits-360-original-size/Validation'

#get number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(test_path + '/*/*.jp*g')

#to get number of folders
folders = glob(train_path + '/*')

#to check images you can use
#plt.imshow(image.loaf_img(np.random.choice(image_files)))
#plt.show()
 
#add preprocessing vgg model but not the top as we'll add our own layer to it
vgg = VGG16(input_shape=IMAGE_SIZE + [3],weights='imagenet',include_top=False)

#don't train the model
for layer in vgg.layers:
    layer.trainable=False

#add our layers
x = Flatten()(vgg.output)

prediction = Dense(len(folders),activation='softmax')(x)

#create model
model = Model(inputs=vgg.input,outputs=prediction)

#compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

#instance of image generator in order to modify the data
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

#get label mapping for confusion matrix plot later
test_gen = gen.flow_from_directory(test_path,target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k,v in test_gen.class_indices.items():
    labels[v] = k
    
#because vgg color palatte is different from the usual keras (BGR instead pf RGB)
for x,y in test_gen:
    print("min:",x[0].min(),'max:',x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break

#create our generators
train_generator = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size,
)

test_generator = gen.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size,
)

#fit model
fit_model = model.fit_generator(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files)//batch_size,
    validation_steps=len(valid_image_files)//batch_size,
)

#create a confusion matrix because we need to pass in arrays and generators
def get_confusion_matrix(data_path,N):
    print("Generating confusion matrix",N)
    predictions = []
    targets = []
    i = 0
    for x,y in gen.flow_from_directory(data_path,target_size=IMAGE_SIZE,shuffle=False,batch_size=batch_size):
        i+=1
        if i&50==0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p,axis=1)
        y = np.argmax(y,axis=1)
        predictions = np.concatenate((predictions,p))
        targets = np.concatenate((targets,y))
        if len(targets)>=N:
            break
    cm = confusion_matrix(targets,predictions)
    return cm


cm = get_confusion_matrix(train_path,len(image_files))
print(cm)
test_cm = get_confusion_matrix(test_path,len(valid_image_files))
print(test_cm)

#plot loss
plt.plot(r.history['loss'],label='train loss')
plt.plot(r.history['val_loss'],label='val loss')
plt.legend()
plt.show()

#plot accuracy
plt.plot(r.history['acc'],label='train_acc')
plt.plot(r.history['val_acc'],label='val acc')
plt.legend()
plt.show()

from util import plot_confusion_matrix
plot_confusion_matrix(cm,labels,title='Train confusion matrix')
plot_confusion_matrix(test_cm,labels,title='Validation confusion matrix')