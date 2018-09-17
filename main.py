# Implementation of the TensforFlow Tutorial for Basic Classification: https://www.tensorflow.org/tutorials/keras/basic_classification 
# author: tfreundo
import tensorflow as tf
# High Level TF-API
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plot

# Parameters for Debugging and tweaking the Learning
debug_plot_exampleImage = False # Plots an example image
debug_plot_trainingdata = False # Plots the first x images from the training data
debug_print_exampleNormalization = False # Prints an example before and after normalization to console
debug_plot_prediction = False # Plots the image and the predictions per class/category
debug_plot_prediction_imgIndex = 1 # The index of the image which should be used for prediction visualization

# MNIST Fashion Dataset (70k grayscale images 28x28pixels with values from 0 to 255 consisting of 10 categories)
datasource = keras.datasets.fashion_mnist

(train_imgs, train_lbls), (test_imgs, test_lbls) = datasource.load_data()

print('Loaded %d samples as Training Data and %d samples as Test Data' %(len(train_imgs), len(test_imgs)))

# The class names corresponding to the labels (0-9)
classnames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


###################################
# Data Overview and Preprocessing #
###################################
def plotSingleImageColorbar(imgData):
    print('Plotting demo image')
    plot.figure()
    plot.imshow(imgData)
    plot.colorbar()
    plot.grid(False)
    plot.show()

def plotImagesWithLabels():
    plot.figure()
    for i in range(25):
        plot.subplot(5,5,i+1)
        plot.xticks([])
        plot.yticks([])
        plot.grid(False)
        plot.imshow(train_imgs[i], cmap=plot.cm.binary)
        plot.xlabel(classnames[train_lbls[i]])
    
    plot.show()

def normalizeData(data):
    norm_data = data / 255.0
    return norm_data


if(debug_plot_exampleImage):
    plotSingleImageColorbar(imgData=train_imgs[0])
if(debug_plot_trainingdata):
    plotImagesWithLabels()

if(debug_print_exampleNormalization):
    print('BEFORE:\n', train_imgs[0])

train_imgs = normalizeData(train_imgs)
test_imgs = normalizeData(test_imgs)
if(debug_print_exampleNormalization):
    print('AFTER NORMALIZATION:\n', train_imgs[0])

######################
# Building the Model #
######################

# Form a sequiental list of layers to execute one after the other (output fed into the next layer)
# Each layer extracts representations from the data fed into them
model = keras.Sequential([
    # Flatten/reformat the input images 2d-array (28x28 pixels) to a 1d array (28*28=784 pixels)
    keras.layers.Flatten(input_shape=(28,28)),
    # Densely resp. fully-connected neural layers
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Returns an array of 10 probabilities (for the 10 categories to predict)
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Configure the model
model.compile(optimizer=tf.train.AdamOptimizer(), # Optimizer to update the model based on data and the according loss
            loss='sparse_categorical_crossentropy', # Loss function measures how accurate the model is during training --> we want to minimize the loss
            metrics=['accuracy']) # Metric that measures/monitors the training/testing steps

######################
# Training the Model #
######################
model.fit(train_imgs, train_lbls, epochs=1)

################################
# Evaluate the Models Accuracy #
################################
test_loss, test_acc = model.evaluate(test_imgs, test_lbls)
print('Test Accuracy = ', test_acc) # It is a hint for overfitting if the accuracy on the test data is less than the accuracy on the training data (possible if e.g. performing to many epochs or reducing the training data)

############################
# Make a single Prediction #
############################

def plotSingleImagePredictionOverview(prediction, predictedLabel, trueLabel, img):
    plot.subplot(1,2,1)
    # Show the image with prediction and probability
    plot.grid(False)
    plot.xticks([])
    plot.yticks([])
    plot.imshow(img, cmap=plot.cm.binary)

    if predictedLabel == trueLabel:
        color = 'green'
    else:
        color = 'red'
    
    plot.xlabel("{} {:2.0f}% ({})".format(predictedLabel,
                                100*np.max(prediction),
                                trueLabel),
                                color=color)

    plot.subplot(1,2,2)
    plot.grid(False)
    plot.xticks(range(10), classnames, rotation=45)
    plot.yticks([])

    plot.bar(range(10), prediction)
    plot.ylim([0,1])

    plot.show()

if(debug_plot_prediction):
    predictions = model.predict(test_imgs) # Make predictions for the whole test data
    prediction = predictions[debug_plot_prediction_imgIndex]
    print('Prediction for Test Image %d:\n' %debug_plot_prediction_imgIndex, prediction)
    prediction_class = np.argmax(prediction)
    print('Predicted class: %s | Actual class: %s' %(classnames[prediction_class], classnames[test_lbls[debug_plot_prediction_imgIndex]]))
    plotSingleImagePredictionOverview(prediction=prediction, predictedLabel=classnames[prediction_class], trueLabel=classnames[test_lbls[debug_plot_prediction_imgIndex]], img=test_imgs[debug_plot_prediction_imgIndex])