import json

# Handle conflicting PyPlot and CV2: from https://stackoverflow.com/questions/38921595/conflicting-opencv-and-matplotlib
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

    # read file containing entries in the database
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
        # Normalize data by dividing value by 255
        colorsElement = [obj[key]['red'] / 255, obj[key]['green'] / 255, obj[key]['blue'] / 255]
        # Add each pixel's rgb value as an element in the array colors
        colors.append(colorsElement)
        # Add label color to the targetLabels array by matching the color of the entry with the color in the colorsArray
        targetLabels.append(colorsArray.index([obj[key]['color']]))

    # Create numpy array xs as an input (tensors) https://www.tensorflow.org/js/guide/layers_for_keras_users
    # Shape (# of images, 3 rgb values)
    x = np.array(colors)

    print(x.shape)

    # print(targetLabels)

    # Convert to numpy array of tensors
    labelTensors = np.array(targetLabels, 'int32')

    # Print labelTensors
    print(labelTensors)

    # y = tf.one_hot(labelTensors, 9, dtype = 'int32')
    # Above didn't work, causing value error as tensor was symbolic
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    # Thus, below worked:
    y = to_categorical(labelTensors, num_classes=9)
    #  Print target output where the target color is 1.0, and the rest is 0.0
    # print(y)
    # Print the shape (# of images, 9 possible colors)
    # print(y.shape)

    # Create a sequential keras model
    # https://keras.io/getting-started/sequential-model-guide/
    model = Sequential()

    # Add fully connected hidden dense layer with 3 inputs from the input layer
    model.add(Dense(7, input_dim=3, name="Hidden_Layer"))
    # Sigmoid activation function, since no categorization is done at this point
    model.add(Activation('sigmoid'))
    # Output layer is fully connected dense layer with nine possible output target (color) values
    model.add(Dense(9, name="Output_Layer"))
    # Softmax activation function since at this point categorization is done.
    # Probability of a particular value (color) is between 0 and 1.
    model.add(Activation('softmax'))

    # Define learning rate
    lr = 0.02
    # Creating an optimizer. Adam optimizer is chosen as with it model trains faster (~ 50 epochs),
    # and accuracy is higher (~ 90%)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Compile the model using above optimizer, and categorical cross entropy as a loss function, since
    # it works well with Softmax activation function and suitable for categorization problems.
    # Accuracy is metrics, to be evaluated by the model during training and testing.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model for 50 epochs, split 10% of the dataset for validation purposes, shuffle the dataset.
    # History variable - to be used for demonstration purposes (Accuracy and loss charts)
    history = model.fit(x, y, epochs=50, validation_split=0.1, shuffle=True, batch_size=32)

    # Plots and the way to show them is taken from here
    # https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
    # Summarize history for Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    # Use nine colors that are not in the dataset to check how the model predicts a color.
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

    
    runNumber = 0
    while True:
        # Check whether it's the first time when GUI is opened
        if runNumber == 0:
            # Close window and stop the program when quit button is clicked
            def quitCommand():
                sys.exit()
            # Continue with the same picture if Continue button is pressed
            def continueCommand():
                root.quit()
            # Select a picture when Choose a Picture button is clicked
            def choosePic():
                root = tk.Tk()
                root.withdraw()

                global file_path
                # previous variable is needed if the user decided to select the picture, but then canceled the operation
                # without selecting a picture
                global previous
                # If some pic is already loaded
                try:
                    file_path
                # If not, and there were no pictures' file paths loaded before, previous variable is an empty string.
                except NameError:
                    previous = ''
                    # print("file_path wasn't defined yet.")
                # If picture is already loaded then the previous variable stores the pic file_path that was before
                else:
                    previous = file_path
                # file_path is selected by the user
                file_path = filedialog.askopenfilename()
                # print(os.path.basename(file_path))
                # If user doesn't select a picture and cancels the operation, the picture remains the same, and the
                # path is taken from the previous variable
                if file_path == '':
                     file_path = previous
                root.quit()

            root = tk.Tk()

            # If the user closes the window.
            def on_closing():
                # Ask the user to confirm and exit
                if messagebox.askokcancel("Quit", "Do you want to quit?"):
                    sys.exit()
            # Set the window title
            root.title("ColorSpotter")
            # Set the size of the window and its location
            root.geometry("400x340+160+370")
            # Set the frame
            frame = Frame(root, relief=RIDGE, borderwidth=2)
            frame.pack(fill=BOTH,expand=1)
            # Set the background
            frame.config(background='light blue')
            # Add text
            label = Label(frame, text="ColorSpotter", bg='light blue',font=('Times 25 bold'))
            label.pack(side=TOP)
            # Set the background picture
            backgroundPicFilename = ImageTk.PhotoImage(Image.open("background.jpg"))
            background_label = Label(frame, image=backgroundPicFilename)
            background_label.pack(side=TOP)
            # Add buttons: Quit, Choose a Picture, Continue
            quit = Button(frame, text='Quit', command=quitCommand)
            quit.pack()
            picture = Button(frame, text='Choose a picture', command=choosePic)
            picture.pack()
            continueButton = Button(frame, text='Continue', state = DISABLED,command=continueCommand)
            continueButton.pack()
        # Set protocol for closing the window
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

        # If file_path for the picture exists
        if file_path != '':
            # imageRGB is an array of RGB values in a selected area of the picture. Obtained by calling a method getRGB()
            imageRGB = getRGB(os.path.basename(file_path))
            # Create a dictionary with colors as keys and number of color occurences as values (initially all zeroes)
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
            # Iterate through each pixel
            for x in range(0, len(imageRGB), 1):
                # Use trained model to predict pixel's color
                prediction = model.predict(np.array([imageRGB[x]]))
                # Select the color with the highest probability and add 1 to the value of the key that corresponds to
                # the same color as the one that was predicted.
                colorsDictionary[colorsArray[prediction.argmax(1)[0]][0]] = colorsDictionary.get(colorsArray[prediction.argmax(1)[0]][0]) + 1

            # Display a dictionary
            print(colorsDictionary)
            # max variable that is responsible for finding the key in the dictionary (colorsDictionary) with the highest value
            max = 0
            # label variable stores the color that was identified
            label = ""
            # Iterate through the dictionary to find the key with the highest value
            for key, value in colorsDictionary.items():
                if value > max:
                    max = value
                    label = key
            # Display the predicted color
            print("The color of the selected area is : " + label)
            messagebox.showinfo("Predicted Color", "The color of the selected area is : " + label)
            # Once the picture is loaded and the file_path exists, Continue button activates to predict the
            # color of the selected area of the same picture
            continueButton.configure(state=NORMAL)
        # Increase the number of runs f the program ('run' corresponds to the button pressed)
        runNumber =+ 1

# Method to select the area of the picture
# Pass image name as a string (file_path)
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

    # Height and width of the selected area
    height, width, channels = imCrop.shape

    # Array of arrays of RGB values of every single pixel in the selected area (to be returned)
    imageRGB = []

    # Iterate through each pixel
    for x in range(0, height, 1):
        for y in range(0, width, 1):
            color = imCrop[x, y]
            # currentColor array to store value as RGB (not BGR as it is in the color variable above)
            currentColor = []
            # Append RGB values in the corect order: R-G-B into the array currentColor (pixel color)
            currentColor.append(color[2])
            currentColor.append(color[1])
            currentColor.append(color[0])
            # Append pixel's RGB color (array) into the main array of RGB values of the selected area (into imageRGB array).
            imageRGB.append(currentColor)
    # Close all the cv2 windows
    cv2.destroyAllWindows()
    # Return the array of arrays of the RGB values of all pixels in the selected area
    return imageRGB

if __name__ == '__main__':
    main()




