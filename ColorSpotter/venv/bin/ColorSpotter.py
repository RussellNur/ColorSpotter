import json
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt

# Array of colors
colorsArray = [['red'], ['orange'], ['yellow'], ['green'], ['blue'], ['purple'], ['brown'], ['gray'], ['pink']]

# read file
with open('colorsclassifier-export-2.json', 'r') as myfile:
    data = myfile.read()

# parse file
obj = json.loads(data)

keys = obj.keys()

# Array colors to store rgb values as elements
colors = []
# Array of labels
targetLabels = []

for key in keys:

    # Normalize data
    colorsElement = [obj[key]['red'] / 255, obj[key]['green'] / 255, obj[key]['blue'] / 255]
    # Add each pixel's rgb value as an element in the array colors
    colors.append(colorsElement)
    # Add label color to the labels array
    #labels.append(colorsArray.index([obj[key]['color']]))
    targetLabels.append(colorsArray.index([obj[key]['color']]))

# Create numpy array xs as an input (tensors) https://www.tensorflow.org/js/guide/layers_for_keras_users
# Shape (# of images, 3 rgb values)
x = np.array(colors)
#print(x)
print(x.shape)
#print(targetLabels)

labelTensors = np.array(targetLabels, 'int32')
print(labelTensors)
#y = tf.one_hot(labelTensors, 9, dtype = 'int32')
# Above didn't work, causing value error as tensor was symbolic
# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
y = to_categorical(labelTensors, num_classes=9)
print(y)
print(y.shape)

# https://keras.io/getting-started/sequential-model-guide/
model = Sequential()
#with tf.name_scope('Hidden_layer') as scope:
#model.add(Dense(36, input_dim=3, name="Hidden_Layer_1"))
#model.add(Activation('sigmoid'))
    #tf.summary.histogram('Hidden_Layer', )
#with tf.name_scope('Target_layer') as scope:
model.add(Dense(7, input_dim=3, name="Hidden_Layer"))
model.add(Activation('sigmoid'))
model.add(Dense(9, name="Output_Layer"))
model.add(Activation('softmax'))
    #tf.summary.histogram('Target_Layer', )

#merged_summaries = tf.summary.merge_all()

#
#saver = tf.train.Saver()

# Define learning rate
lr = 0.02

# Creating an optimizer
#with tf.name_scope('train') as scope:
#optimizer = keras.optimizers.Adam(lr=lr)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)#.minimize(cost)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x, y, epochs=1000, validation_split=0.1, shuffle=True, batch_size=32)

# Plots and the way to show them is taken from here
# https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#pyplot.plot(history.history['accuracy'])
#pyplot.show()

prediction1 = model.predict(np.array([[0, 204, 102]]))
print('Should be Green. Predicted:')
print(colorsArray[prediction1.argmax(1)[0]])
prediction2 = model.predict(np.array([[255, 51, 51]]))
print('Should be Red. Predicted:')
print(colorsArray[prediction2.argmax(1)[0]])
prediction3 = model.predict(np.array([[51, 51, 255]]))
print('Should be Blue. Predicted:')
print(colorsArray[prediction3.argmax(1)[0]])
prediction4 = model.predict(np.array([[255, 102, 255]]))
print('Should be Pink. Predicted:')
print(colorsArray[prediction4.argmax(1)[0]])
prediction5 = model.predict(np.array([[255, 255, 51]]))
print('Should be Yellow. Predicted:')
print(colorsArray[prediction5.argmax(1)[0]])
prediction6 = model.predict(np.array([[224, 224, 224]]))
print('Should be Grey. Predicted:')
print(colorsArray[prediction6.argmax(1)[0]])
prediction7 = model.predict(np.array([[102, 51, 0]]))
print('Should be Brown. Predicted:')
print(colorsArray[prediction7.argmax(1)[0]])
prediction8 = model.predict(np.array([[76, 0, 153]]))
print('Should be Purple. Predicted:')
print(colorsArray[prediction8.argmax(1)[0]])
prediction9 = model.predict(np.array([[255, 153, 51]]))
print('Should be Orange. Predicted:')
print(colorsArray[prediction9.argmax(1)[0]])
