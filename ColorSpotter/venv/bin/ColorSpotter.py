import json

import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

from matplotlib import pyplot as plt

import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Button
from tkinter import mainloop
from PIL import ImageTk, Image

import os
import sys

cv2.ocl.setUseOpenCL(False)



def main():


    # Array of colors
    colorsArray = [['red'], ['orange'], ['yellow'], ['green'], ['blue'], ['purple'], ['brown'], ['gray'], ['pink']]

    # read file
    with open('colorsclassifier-export-3.json', 'r') as myfile:
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
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x, y, epochs=50, validation_split=0.1, shuffle=True, batch_size=32)

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
    # # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


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

    #file_path = ''
    count = 0
    while True:
        if count == 0:
            def quitCommand():
                sys.exit()

            def continueCommand():
                root.quit()

            def choosePic():
                root = tk.Tk()
                root.withdraw()
                global file_path
                global previous
                # print(file_path)
                try:
                    file_path
                except NameError:
                    previous = ''
                    print("file_path wasn't defined yet.")
                else:
                    previous = file_path
                file_path = filedialog.askopenfilename()
                print(os.path.basename(file_path))
                if file_path == '':
                     file_path = previous
                root.quit()

            root = tk.Tk()
            def on_closing():
                if messagebox.askokcancel("Quit", "Do you want to quit?"):
                    sys.exit()
            root.title("ColorSpotter")
            root.geometry("400x340+160+370")
            frame = Frame(root, relief=RIDGE, borderwidth=2)
            frame.pack(fill=BOTH,expand=1)
            frame.config(background='light blue')
            label = Label(frame, text="ColorSpotter", bg='light blue',font=('Times 25 bold'))
            label.pack(side=TOP)
            backgroundPicFilename = ImageTk.PhotoImage(Image.open("background.jpg"))
            background_label = Label(frame, image=backgroundPicFilename)
            background_label.pack(side=TOP)
            quit = Button(frame, text='Quit', command=quitCommand)
            quit.pack()
            picture = Button(frame, text='Choose a picture', command=choosePic)
            picture.pack()
            continueButton = Button(frame, text='Continue', state = DISABLED,command=continueCommand)
            continueButton.pack()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()


        if file_path != '':
            imageRGB = getRGB(os.path.basename(file_path))
            colorsDictionary = {
                "red": 0,
                "orange": 0,
                "yellow": 0,
                "green": 0,
                "blue": 0,
                "purple": 0,
                "brown": 0,
                "gray": 0,
                "pink": 0
            }

            for x in range(0, len(imageRGB), 1):
                #print("LOOK: ")
                #print(imageRGB[x])
                prediction = model.predict(np.array([imageRGB[x]]))

                #print("Color predicterd:")
                #print(colorsArray[prediction.argmax(1)[0]])
                colorsDictionary[colorsArray[prediction.argmax(1)[0]][0]] = colorsDictionary.get(colorsArray[prediction.argmax(1)[0]][0]) + 1

            #print(colorsArray[prediction.argmax(1)[0]])
            print(colorsDictionary)
            max = 0
            label = ""
            for key, value in colorsDictionary.items():
                if value > max:
                    max = value
                    label = key
            print("The color of the selected area is : " + label)
            messagebox.showinfo("Predicted Color", "The color of the selected area is : " + label)
            continueButton.configure(state=NORMAL)

        count =+ 1


def getRGB(imageName):
    """

    :type imageName: str
    """
    # opencv tutorial is from here: https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
    # Read image
    im = cv2.imread(imageName)

    # Select ROI
    r = cv2.selectROI("ColorSpotter - select a rectangle", im)

    # Crop image
    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    height, width, channels = imCrop.shape

    imageRGB = []

    for x in range(0, height, 1):
        for y in range(0, width, 1):
            color = imCrop[x, y]
            currentColor = []
            # print(color[0], color[1], color[2])
            currentColor.append(color[2])
            currentColor.append(color[1])
            currentColor.append(color[0])
            imageRGB.append(currentColor)
    cv2.destroyAllWindows()
    #print(imageRGB)
    return imageRGB

if __name__ == '__main__':
    main()




