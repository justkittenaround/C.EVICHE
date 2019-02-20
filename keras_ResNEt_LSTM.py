###ResNet50_LSTM_Keras



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import keras
from keras.applications import resnet50
from keras.models import Sequential

#hyperparameters
epochs = 3
batch_size = 1

##model
model = Sequential()
model.add(resnet50.ResNet50(weights='imagenet', include_top=False, input_shape='256,256,1'))
model.add(model.keras.layers.LSTM(1, activation=‘relu’, recurrent_activation=‘relu’)
model.compile(loss='categorical_crossentropy',
              optimizer=‘adam’,
              metrics=['accuracy'])

model = model()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
##train
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

##evaluate
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)

##test
classes = model.predict(x_test, batch_size=batch_size)
