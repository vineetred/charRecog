
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import seaborn as sb
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras import models
from keras.preprocessing import image
from time import time


# In[8]:


#Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)
test_detagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

training_set = train_datagen.flow_from_directory('/home/vineetred/set/English/Img/GoodImg/Bmp/',
                                                 target_size = (32,
                                                 32),
                                                 batch_size = 32,
                                                 color_mode = 'grayscale',
                                                 class_mode =
                                                     'categorical', subset = "training")

validation = train_datagen.flow_from_directory('/home/vineetred/English/Img/GoodImg/Bmp/',
                                                 target_size = (32,
                                                 32),
                                                 batch_size = 1,
                                          
                                                 class_mode = 
                                                     'categorical', color_mode = 'grayscale',subset = 'validation',shuffle=False)

'''
Using the two datasets below just for testing. Ignore them.
'''
collateral = test_detagen.flow_from_directory('/home/vineetred/set/English/Img/GoodImg/Bmp/',
                                                 target_size = (32,
                                                 32),
                                                 batch_size = 32,
                                                 color_mode = 'grayscale',
                                                 class_mode =
                                                     'categorical', subset = "training")

testing_set = test_detagen.flow_from_directory('/home/vineetred/English/Img/GoodImg/Bmp/',
                                                 target_size = (32,
                                                 32),
                                                 batch_size = 1,
                                          
                                                 class_mode = 
                                                     None, color_mode = 'grayscale',subset = 'validation',shuffle=False)

#In case the model does not increase validation
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              
                              patience=4,
                              verbose=1, mode='auto')


# In[3]:


#My CNN
#Load in case you do not want to train the model again. Make sure to comment everything else out!
# model = load_model('my_model.h5')
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(32,32,1)))
model.add(MaxPooling2D(pool_size =(2,2), strides = (1,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size =(2,2), strides = (1,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size =(2,2), strides = (1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size =(2,2), strides = (1,1)))

model.add(Flatten())
model.add(Dropout(0.40))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))

model.add(Dense(62,activation="softmax"))
model.summary()


# In[4]:


model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


# In[9]:


model.fit_generator(training_set,
                    steps_per_epoch = 299,
                    epochs = 50,
                    validation_data=validation,
                    validation_steps=1899,
                    callbacks=[early_stop, tensorboard],
                    use_multiprocessing = True,
                    workers = 20)


# In[10]:


#TESTING/EVALUATION
model.evaluate_generator(validation, steps = 1899)


# In[23]:


#Confusion Matrix

predictions = model.predict_generator(validation, steps = 1899) #Stores the predictions
predictions = np.argmax(predictions,axis = 1) #Only keeps the most probable class
true_predictions = validation.classes #Stores true classes
cm = confusion_matrix(true_predictions, predictions) #Creates a confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #Normalises to float
sb.heatmap(cm_normalized,cmap='viridis')


# In[15]:


#Predict images classes

img_path = '/home/vineetred/set/English/Img/GoodImg/Bmp/Sample009/img009-00009.png' #Path to the image

#Normalising the image
img_arr = image.load_img(img_path,target_size=(32,32),color_mode='grayscale')
img_arr = image.img_to_array(img_arr)
img_arr /= 255.
img_arr = np.expand_dims(img_arr, axis=0)

 

#Displaying the image
img = plt.imread(img_path)
plt.imshow(img)


model.predict_classes(img_arr)
# Returns a list of five Numpy arrays: one array per layer activation


# In[ ]:


#Seeing what the convolution layer sees

# Extracts the outputs of the top 12 layers
outputLayers = [layer.output for layer in model.layers[:12]] 

#Passing the image to the conv layer
activation_model = models.Model(inputs=model.input, outputs=outputLayers) 
activations = activation_model.predict(img_arr)

layerNumber = 7 #Define which layer you want to look at
first_layer_activation = activations[layerNumber]

print(first_layer_activation.shape)

for i in range(0,32):
    plt.matshow(first_layer_activation[0, :, :, i], cmap='viridis')

