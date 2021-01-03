# import required python libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras.models import Input, Model, model_from_json, Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras import optimizers
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline


 # use GPU for faster training
 physical_devices = tf.config.experimental.list_physical_devices('GPU')
 print('Available Number of GPU:', len(physical_devices))
 tf.config.experimental.set_memory_growth(physical_devices[0], True)
 
 # Load and preprocess data
# This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. See more info at the CIFAR homepage.

#x_train, x_test: uint8 arrays of RGB image data with shape (num_samples, 32, 32, 3) if the data format is 'channels_last'.
#y_train, y_test: uint8 arrays of category labels (integers in range 0-9) each with shape (num_samples, 1). 
#For AutoEncoder we dont need y_train and y_test.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Split 60000 using 80,10,10
# 48000 train , 6000 validation, 6000 test
X = np.concatenate((x_train, x_test), axis=0)
print(X.shape)

X_train, X_test = train_test_split(X, test_size=0.1, random_state=1)
X_train, X_val = train_test_split(X_train, test_size=0.111111, random_state=1)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
# Normalize the data
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255
X_test = X_test.astype('float32') / 255

  """
  Create a CAE:
  """
input_shape = (32, 32, 3)
  
model = Sequential()
  # L1 Convolution layer with window size of (3,3):
model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape, name='conv_1'))
  # L2 MaxPool:
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
  # L3 Convolution layer with window size of (3,3):
model.add(Conv2D(filters=12, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape, name='conv_2'))
  # L4 MaxPool:
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
  # L5 Convolution layer with window size of (3,3): Latent Space Represantation
model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape, name='conv_3'))
  # L6 Upsample 2x2
model.add(UpSampling2D(size=(2, 2), interpolation="nearest"))
  # L7 Convolution layer with window size of (3,3):
model.add(Conv2D(filters=12, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape, name='conv_4'))
  # L8 Upsample 2x2
model.add(UpSampling2D(size=(2, 2), interpolation="nearest"))
  # L9 Convolution layer with window size of (3,3):
model.add(Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='softmax', input_shape=input_shape, name='output'))

model.summary()

 # Run CAE
lowest_loss = 1
model_json = []
def run_CAE(my_model, batch_size = 48, epochs = 20, lowest_loss=lowest_loss, json_string=model_json):
    model = my_model
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

  # Plot the evolution of the error with epochs"
    training = model.fit(X_train, X_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         steps_per_epoch = 1000,
                         verbose=0,
                         validation_data=(X_val, X_val))
  
    score = model.evaluate(X_test, X_test, verbose=0)
    print('Test loss:', score[0])
    loss = score[0]
    
    if loss < lowest_loss:
      model.save('best_model.h5')
      json_string = model.to_json()
      lowest_loss = loss
    
    return training, loss, lowest_loss, json_string
    
      trainning,_,_,_ = run_CAE(model)

  # Loss
plt.plot(trainning.history['loss'])
plt.plot(trainning.history['val_loss'])
plt.title('the loss with epochs')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()
  
  # Accuracy
plt.plot(trainning.history['accuracy'])
plt.plot(trainning.history['val_accuracy'])
plt.title('the accuracy with epochs')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()
def build_CAE(input_shape, strides, kernel_size, layer_filters):
  # Encoder
  input = Input(shape=input_shape, name='Input_layer')
  x = input
  for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               activation='relu')(x)
  # Decoder
  for filters in layer_filters[::-1]:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               activation='relu')(x)
    if strides != 1:
      x = UpSampling2D(size=(2,2))(x)

  output = Conv2D(filters=3,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               activation='relu',
               name='Output_layer')(x)
  # CAE
  autoencoder = Model(input, output)
  
  return autoencoder
  
  
input_shape = (32, 32, 3) 
batch_size = 48
kernel_size = 3
strides = 1
layer_filters_sets = [[8,16,32],[32,64,128]]
loss_set = []
  
for layer_filters in layer_filters_sets:
  
  print('When strides =', strides)

  model = build_CAE(input_shape, strides, kernel_size, layer_filters)

  model.summary()
  
  trainning, test_loss,_,_ = run_CAE(my_model = model, batch_size = 48, epochs = 10, lowest_loss=lowest_loss, json_string=model_json)

  plt.plot(trainning.history['loss'], label=('layer_filters=' , layer_filters))
  
  loss_set.append(test_loss)

  print('*' * 60)

plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')

# produce LAB images
from skimage.color import rgb2gray, gray2rgb, rgb2lab

lab_train = rgb2lab(X_train)
X_lab_train = lab_train[:,:,:,0].reshape(lab_train.shape[0], 32, 32, 1)
y_lab_train = lab_train[:,:,:,1:]/128

lab_test = rgb2lab(X_test)
X_lab_test = lab_test[:,:,:,0].reshape(lab_test.shape[0], 32, 32, 1)
y_lab_test = lab_test[:,:,:,1:]/128

lab_val = rgb2lab(X_val)
X_lab_val = lab_val[:,:,:,0].reshape(lab_val.shape[0], 32, 32, 1)
y_lab_val = lab_val[:,:,:,1:]/128

input_shape = (32,32,1)
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu', input_shape=input_shape, name='conv_1'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same', activation='relu',  name='conv_4'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu',  name='conv_5'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',  name='conv_6'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu',  name='conv_7'))
model.add(UpSampling2D(size=(2, 2), interpolation="nearest"))
model.add(Conv2D(filters  =128, kernel_size=(3, 3), strides=1, padding='same', activation='relu',  name='conv_8'))
model.add(UpSampling2D(size=(2, 2), interpolation="nearest"))
model.add(Conv2D(filters=2, kernel_size=(3, 3), strides=1, padding='same', activation='tanh', name='output'))
model.add(UpSampling2D(size=(2, 2), interpolation="nearest"))
model.summary()
  
model.compile(loss=keras.losses.mean_squared_error,
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                metrics=['accuracy'])

model.fit(X_lab_train, y_lab_train, batch_size=48, validation_data=(X_lab_val, y_lab_val), steps_per_epoch=1000 ,epochs=10, verbose=2)

output = model.predict(X_lab_test)
pre_img = np.zeros((len(X_lab_test), 32, 32, 3))
pre_img[:,:,:,0] = X_lab_test[:,:,:,0]
pre_img[:,:,:,1:] = output*128

from skimage.color import lab2rgb
img_n = np.random.randint(len(X_test))
origin_img = X_test[img_n]
fig, axs =plt.subplots(1,2)
axs[0].imshow(origin_img)
axs[1].imshow(lab2rgb(pre_img[img_n]))
plt.show()
