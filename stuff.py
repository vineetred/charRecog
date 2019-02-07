import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import glob

images = []
labels = []
for path in glob.glob("/home/hotshot2797/English/img/GoodImg/Bmp/Sample*"):
#     print(path)
#     for img in glob.glob("/Users/vineet/Desktop/cnn/English/img/GoodImg/Bmp/Sample001/*.png"):
    for img in glob.glob(path+"/*.png"):
#         print(img)
        n= plt.imread(img)
        images.append(np.resize(n,(32,32,3)))
        prefix = img.split("/Sample")
        prefix = prefix[1].split("/")
        labels.append(prefix[0])

image2 = np.array(images)
image2.shape


labels = np.array(labels)
labels.shape

from keras.utils import to_categorical
labels = to_categorical(labels)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
#create model
model = Sequential()
#add model layers
model.add(Conv2D(filters = 32, kernel_size=(4,4), activation="relu",input_shape = (32,32,3)))
model.add(Conv2D(64, kernel_size=(4,4), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(4,4), activation="relu"))
model.add(Conv2D(10, kernel_size=(4,4), activation="relu"))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(63, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(image2, labels, epochs=5)