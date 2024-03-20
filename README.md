# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![Screenshot 2024-03-20 105922](https://github.com/vinodkumar-s/mnist-classification/assets/113497226/84866384-2009-4066-8286-59de278050de)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset

### STEP 3:
Scale the dataset between it's min and max values

### STEP 4:
Using one hot encode, encode the categorical values

### STEP 5:
Split the data into train and test

### STEP 6:
Build the convolutional neural network model

### STEP 7:
Train the model with the training data

### STEP 8:
Plot the performance plot

### STEP 9:
Evaluate the model with the testing data

### STEP 10:
Fit the model and predict the single input


## PROGRAM

### Name: VINOD KUMAR S
### Register Number: 212222240116

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=16, kernel_size=(9,9), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(3,3)))
model.add(layers.Flatten())
model.add(layers.Dense(65,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=64,validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

print("212222240116 VINOD KUMAR S")
metrics[['accuracy','val_accuracy']].plot()

print("212222240116 VINOD KUMAR S")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("212222240116 VINOD KUMAR S")
print(confusion_matrix(y_test,x_test_predictions))

print("212222240116 VINOD KUMAR S")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('6.png')

img = image.load_img('6.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-03-16 093441](https://github.com/vinodkumar-s/mnist-classification/assets/113497226/3bb37fe9-43f3-4012-be6f-10eaf1f1735c)

![Screenshot 2024-03-16 093513](https://github.com/vinodkumar-s/mnist-classification/assets/113497226/10d7ab6b-da4f-489e-9ea4-3f24a502ea44)


### Classification Report

![Screenshot 2024-03-16 093604](https://github.com/vinodkumar-s/mnist-classification/assets/113497226/e95665f9-547e-4993-8d1c-6cde1ba5fb77)

### Confusion Matrix

![Screenshot 2024-03-16 093542](https://github.com/vinodkumar-s/mnist-classification/assets/113497226/d67dac81-d56c-403a-93c8-fff37585ccb7)

### New Sample Data Prediction

![Screenshot 2024-03-16 093638](https://github.com/vinodkumar-s/mnist-classification/assets/113497226/6787873d-e78c-475f-83ac-c63f7e28e3ac)

![Screenshot 2024-03-16 093654](https://github.com/vinodkumar-s/mnist-classification/assets/113497226/83d60152-bf73-4383-8f27-606931b2ca29)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
