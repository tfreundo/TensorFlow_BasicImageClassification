# TensorFlow_BasicImageClassification
An implementation of the TensorFlow Tutorial for BasicClassification using the MNIST Fashion Dataset.
The original tutorial can be found [here](https://www.tensorflow.org/tutorials/keras/basic_classification).

# Debugging and Visualization
You can choose from multiple debugging and visualization options which fulfill the purpose of learning and understanding what happens after some important steps.
```
debug_plot_exampleImage = False # Plots an example image
debug_plot_trainingdata = False # Plots the first 25 images from the training data
debug_print_exampleNormalization = False # Prints an example before and after normalization to console
debug_plot_prediction = False # Plots the image and the predictions per class/category
debug_plot_prediction_imgIndex = 1 # The index of the image which should be used for prediction visualization
```

## debug_plot_exampleImage
This option will plot the first image of the Training Dataset as Colorbar (28x28 pixels with color encoding from 0 to 255)
![debug_plot_exampleImage](/images/debug_plot_exampleImage.png)

## debug_plot_trainingdata
This option will print the first 25 images in a 5x5 plot.
![debug_plot_trainingdata](/images/debug_plot_trainingdata.png)

## debug_print_exampleNormalization
This option will print the first image data vector before and after normalization to the console.

## debug_plot_prediction
This option will plot the image (index of the image in the Test Dataset defined by **debug_plot_prediction_imgIndex**) and the predicted probabilities for each of the 10 classes/categories.
The prediction text will be green if the expected class equals the actual class and red if they differ.
The text below the image 
![debug_plot_prediction](/images/debug_plot_prediction.png)
