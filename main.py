
import tensorflow as tf
import matplotlib as matplotlib
import matplotlib_inline
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from PIL import Image
from numpy import array, int64

np.random.seed(11) # It's my lucky number
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import itertools



import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications.resnet import ResNet50
from keras import backend as K
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Flatten, add, Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy

from keras.layers import Conv2D, MaxPool2D

# importing test and train folders

folder_benign_train = 'C:/Users/kajak/Desktop/Studia/Semestr 5/projekt Bioc/projektBioca/projektBioca-main/data/train/benign'

#'C:/Users/werka/OneDrive/Pulpit/SEM5/bioca projekcik/ProjektBioca/ProjektBioca/archive/data/train/benign'

folder_malignant_train = 'C:/Users/kajak/Desktop/Studia/Semestr 5/projekt Bioc/projektBioca/projektBioca-main/data/train/malignant'
#'C:/Users/werka/OneDrive/Pulpit/SEM5/bioca projekcik/ProjektBioca/ProjektBioca/archive/data/train/malignant'

folder_benign_test = 'C:/Users/kajak/Desktop/Studia/Semestr 5/projekt Bioc/projektBioca/projektBioca-main/data/test/benign'
#'C:/Users/werka/OneDrive/Pulpit/SEM5/bioca projekcik/ProjektBioca/ProjektBioca/archive/data/test/benign'

folder_malignant_test = 'C:/Users/kajak/Desktop/Studia/Semestr 5/projekt Bioc/projektBioca/projektBioca-main/data/test/malignant'
#'C:/Users/werka/OneDrive/Pulpit/SEM5/bioca projekcik/ProjektBioca/ProjektBioca/archive/data/test/malignant'

folder_benign_validation="C:/Users/kajak/Desktop/Studia/Semestr 5/projekt Bioc/projektBioca/projektBioca-main/validation/benign"
#"C:/Users/werka/OneDrive/Pulpit/SEM5/bioca projekcik/ProjektBioca/ProjektBioca/archive/validation/benign"

folder_malignant_validation="C:/Users/kajak/Desktop/Studia/Semestr 5/projekt Bioc/projektBioca/projektBioca-main/validation/malignant"
#"C:/Users/werka/OneDrive/Pulpit/SEM5/bioca projekcik/ProjektBioca/ProjektBioca/archive/validation/malignant"

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# Load in training pictures
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]
X_malignant = np.array(ims_malignant, dtype='uint8')


# Load in testing pictures
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')

# Load in validation pictures
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign_validation = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]
X_malignant_validation = np.array(ims_malignant, dtype='uint8')


# Create labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])

y_benign_validation=np.zeros(X_benign_validation.shape[0])
y_malignant_validation=np.ones(X_malignant_validation.shape[0])

# Merge data
X_train = np.concatenate((X_benign, X_malignant), axis = 0)
y_train = np.concatenate((y_benign, y_malignant), axis = 0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)

X_val= np.concatenate((X_benign_validation, X_malignant_validation), axis = 0)
y_val= np.concatenate((y_benign_validation, y_malignant_validation), axis = 0)

print(f'Zbiór uczący: {X_benign.shape,X_malignant.shape}, zbiór testowy: {X_benign_test.shape,X_malignant_test.shape}, zbiór walidacyjny:{X_benign_validation.shape,X_malignant_validation.shape}')

X_train = np.vstack([X_benign.astype('float32')/255.0, X_malignant.astype('float32') / 255.0])
X_val = np.vstack([X_benign_validation.astype('float32')/255.0, X_malignant_validation.astype('float32') / 255.0])
X_test =np.vstack([X_benign_test.astype('float32')/255.0, X_malignant_test.astype('float32') / 255.0])


def draw_curves(history, key1 ='accuracy', ylim1=(0.7, 1.00), key2='loss', ylim2=(0.0, 0.6)):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')
    plt.show()


X_train_cnn = X_train.reshape((X_train.shape[0], 224, 224, 3))
X_test_cnn = X_test.reshape((X_test.shape[0], 224, 224, 3))
X_val_cnn = X_val.reshape((X_val.shape[0], 224, 224, 3))


#CNN network -first model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train_cnn, y_train, epochs=50, verbose=1, validation_data = (X_val_cnn, y_val), callbacks = [EarlyStopping(monitor='val_loss',
                              patience=5,
                              verbose=1)],)

draw_curves(history, key1='accuracy', key2='loss')
score = model.evaluate(X_val_cnn, y_val, verbose=0)
print("CNN Error: %.2f%%" % (100-score[1]*100))


####
y_pred = model.predict(X_test)

y_pred_binary = np.where(y_pred[:, 0] > y_pred[:, 1], 0, 1)


confusion_matrix=confusion_matrix(y_test,y_pred_binary)

print(confusion_matrix)

plt.imshow(confusion_matrix, cmap="Blues", interpolation="nearest")

# Add label to squares in matrix
for i in range(2):
    for j in range(2):
        text = plt.text(j, i, confusion_matrix[i][j], ha="center", va="center", color="black")

# add labels to x and y axis
plt.xticks([0, 1], ["Predicted: positive", "Predicted: negative"])
plt.yticks([0, 1], ["True: positive", "True: negative"])

# Add label to plot
plt.title("Confusion matrix")

# Show plot
plt.show()


# calculate accuracy
Accuracy = accuracy_score(y_test, y_pred_binary)

# calculate precision
Precision = precision_score(y_test, y_pred_binary)

# calculate recall
Recall = recall_score(y_test, y_pred_binary)

# calculate specificity
Specificity = confusion_matrix[0][0]/(confusion_matrix[1][1]+confusion_matrix[0][0])

# calculate F1 score
F1_score = f1_score(y_test, y_pred_binary)

# print metrics
print("Accuracy:", Accuracy)
print("Precision:", Precision)
print("Recall:", Recall)
print("Specificity:", Specificity)
print("F1 score:", F1_score)

'''
##RESNET50
from keras.applications.resnet import ResNet50
from keras.utils import to_categorical

# Create a ResNet50 model with pre-trained weights
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers of the pre-trained model
for layer in resnet.layers:
    layer.trainable = False

# Create new layers on top of the pre-trained model
x = resnet.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

# Build the new model
model = Model(inputs=resnet.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
y_val = to_categorical(y_val, num_classes=2)
y_train = to_categorical(y_train, num_classes=2)
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# load the ResNet50 model
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# add a new top layer
x = resnet.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# create a new model with the new top layer
model = Model(inputs=resnet.input, outputs=predictions)

# mark all layers as non-trainable
for layer in resnet.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load the data
# Replace X with your image data and Y with your labels
X_train, X_val, Y_train, Y_val = train_test_split(X_val, y_val, test_size=0.2)

# create data generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# fit the data generator on the training data
train_datagen.fit(X_train)

# fine-tune the model
model.fit(train_datagen.flow(X_train, Y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=10)'''



