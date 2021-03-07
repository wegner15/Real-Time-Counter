import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# Helper libraries
import numpy as np
import matplotlib.pyplot as pl

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import pandas as pd


import warnings
warnings.filterwarnings("ignore")
import tensorflow.keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model,load_model



test_size=37
batch_size=12
epochs=4
train_path='C:/Users/Bogingo/Desktop/Product'
test_path='C:/Users/Bogingo/Desktop/Product Test/Test'
#Beginning of image loading
conv_base =  InceptionV3(weights='imagenet',include_top=False,
                         input_shape=(300, 300, 3))
output = conv_base.layers[-1].output
output = tensorflow.keras.layers.Flatten()(output)
model_tl = Model(conv_base.input, output)
model_tl.trainable = False
for layer in model_tl.layers:
    layer.trainable = False
layers = [(layer, layer.name, layer.trainable) for layer in
               model_tl.layers]
model_layers=pd.DataFrame(layers, columns=['Layer Type', 'Layer  Name', 'Layer Trainable'])
print(model_layers)


target_size=(300,300) #resize all images to 300x300
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3,
                                   rotation_range=50,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   brightness_range = [0.8, 1.2],
                                   fill_mode='nearest',
                                   validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
# The list of classes will be automatically inferred from the subdirectory names/structure under train_dir
train_generator = train_datagen.flow_from_directory(
                  train_path,
                  target_size=target_size,#
                  batch_size=batch_size,
                  class_mode='categorical',
                  subset='training')
validation_generator = train_datagen.flow_from_directory(
                       train_path,
                       target_size=target_size,
                       batch_size=batch_size,
                       class_mode='categorical',
                       subset='validation')
# building a linear stack of layers with the sequential model
model =Sequential()
model.add(model_tl)
# hidden layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# output layer
model.add(Dense(5, activation='softmax'))
# compiling the sequential model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
print(model.summary())
#End of data loading
#_______________________________________________________________
#Beginning of training
from tensorflow.keras.callbacks import *
filepath='./MyCNN'
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False,
                             mode='max')
callbacks_list = [checkpoint]

history = model.fit(
          train_generator,
          steps_per_epoch=train_generator.samples//batch_size,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples//batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          callbacks=callbacks_list)
# Model evaluation
scores_train = model.evaluate(train_generator,verbose=1)
scores_validation = model.evaluate(validation_generator,verbose=1)
print("Train Accuracy: %.2f%%" % (scores_train[1]*100))
print("Validation Accuracy: %.2f%%" % (scores_validation[1]*100))
#For plotting Accuracy and Loss

def LearningCurve(history):
# summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
LearningCurve(history)
#Save the trained model to a file

model_weight_file=r'C:\Users\Bogingo\PycharmProjects\imageClassification\MyCNN"Product_Classification_Model.h5'
#model.save(model_weight_file)
#End of learning 
#__________________________________________________________________
#Begining of testing 
# We '''take the ceiling because we do not drop the remainder of the batch
'''
compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
test_steps = compute_steps_per_epoch(test_size)
test_generator = test_datagen.flow_from_directory(
                 test_path,
                 target_size=target_size, 
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
test_generator.reset()
#Calling the saved model for making predictions
tl_img_aug_cnn = load_model(model_weight_file)
pred=tl_img_aug_cnn.predict(test_generator,
                            verbose=1,
                            steps=test_steps)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
#create a function for visualizing model performance

import seaborn as sns
def PerformanceReports(conf_matrix,class_report,labels):
    ax= plt.subplot()
    sns.heatmap(conf_matrix, annot=True,ax=ax)
    #labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()
    ax= plt.subplot()
    sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T,  
                annot=True,ax=ax)
    ax.set_title('Classification Report')
    plt.show()
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
labels=['Carrier_With_Product','Feeds','Raw Materials','Wheat Flour Brown', 'Wheat Flour-White']
test_labels = [fn.split('/')[0] for fn in filenames]
cm=confusion_matrix(test_labels,predictions)
print(cm)
cr=classification_report(test_labels, predictions)
class_report=classification_report(test_labels, predictions,
                                   target_names=labels,
                                   output_dict=True)
print(cr)
PerformanceReports(cm,class_report,labels)'''
'''
img_path = "/content/drive/MyDrive/Google Photos/2018/09/IMG-20180901-WA0022.jpg"
pl.imshow(image.load_img(img_path, target_size=(300, 300)))
pl.show()
from keras.models import load_model


model = load_model(model_weight_file)

class_names = ['Carrier_With_Products', 'Feeds', 'Raw Materials', 'Wheat Flour Brown',
               "Wheat Flour White"]

img = image.load_img(img_path , target_size=(300, 300))
img = np.array(img)
img = img / 255.0
img = img.reshape(1,300,300,3)
predictions = model.predict(img)
prediction_result = np.argmax(predictions[0])
prediction_Confidence=predictions[0][prediction_result]

if (prediction_Confidence>0.6):
  print("Predicted Class:",class_names[prediction_result])
  print("Prediction Confidence",prediction_Confidence)
  print("All Classes:")
  for x in range (5):
    print(class_names[x],":", predictions[0][x])
else:
  print("Item does not belong to any of the classes")
'''
