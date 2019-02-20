import keras
from keras.applications import resnet50



##model
resnet_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape='256,256,1')


##train


##test
predictions = resnet50.predict(x)
