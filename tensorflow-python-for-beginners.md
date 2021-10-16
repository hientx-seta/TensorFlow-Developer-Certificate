* Code labs: [Introduce Python](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb#scrollTo=vIgmFZq4zszl) - [Video](https://www.youtube.com/watch?v=xp7DGVGf8_c)

* Code labs [The Basics: Training Your First Model](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb#scrollTo=gg4pn6aI1vms) - [Video](https://www.youtube.com/watch?v=o7U-ELsI0FE&t=3s)

* Code labs [Classifying Images of Clothing](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb#scrollTo=jYysdyb-CaWM) - [Video](https://www.youtube.com/watch?v=o7U-ELsI0FE)

* Code labs [Image Classification with Convolutional Neural Networks](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l04c01_image_classification_with_cnns.ipynb#scrollTo=jYysdyb-CaWM) - [Video](https://www.youtube.com/watch?v=niylIkhErZo)

* Video [What is Machine Learning?](https://www.youtube.com/watch?v=UxKbUwj5hmU&t=106s)

* Video [Dense layer](https://www.youtube.com/watch?v=lYC2rHBYcCI&t=2s)

* Video [Neural Network](https://www.youtube.com/watch?v=kwiMF2XH0T0)

* [Rectified Linear Units (ReLU) in Deep Learning](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning)

* Video [Convolutions](https://www.youtube.com/watch?v=sAPg-qaT0b4)

* Video [Max Pooling](https://www.youtube.com/watch?v=o_DJ-FO6dw0)

* [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)


### Some Machine Learning terminology
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)

fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

 - **Feature** — The input(s) to our model. In this case, a single value — the degrees in Celsius.

 - **Labels** — The output our model predicts. In this case, a single value — the degrees in Fahrenheit.

 - **Example** — A pair of inputs/outputs used during training. In our case a pair of values from `celsius_q` and `fahrenheit_a` at a specific index, such as `(22,72)`.
 - **Layer** — A collection of nodes connected together within a neural network.
 - **Model** — The representation of your neural network
 - **Dense and Fully Connected (FC)** : Each node in one layer is connected to each node in the previous layer.
 - **Weights and biases** : The internal variables of model
 - **Loss** : The discrepancy between the desired output and the actual output
 - **MSE** : Mean squared error, a type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.
 - **Gradient Descent** : An algorithm that changes the internal variables a bit at a time to gradually reduce the loss function.
 - **Optimizer** : A specific implementation of the gradient descent algorithm. (There are many algorithms for this. In this course we will only use the “Adam” Optimizer, which stands for ADAptive with Momentum. It is considered the best-practice optimizer.)
 - **Learning rate** : The “step size” for loss improvement during gradient descent.
 - **Batch** : The set of examples used during training of the neural network
 - **Epoch** : A full pass over the entire training dataset
 - **Forward pass** : The computation of output values from input
 - **Backward pass (backpropagation)** : The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.
 - **Flattenings** : The process of converting a 2d image into 1d vector
 - **ReLUs** : An activation function that allows a model to solve nonlinear problems
 - **Softmax** : A function that provides probabilities for each possible output class
 - **Classification** : A machine learning model used for distinguishing among two or more output categories
 - **The Conv2D layer** : also has kernels (filters) whose values need to be tuned as well. So, in a Conv2D layer the values inside the filter matrix are the variables that get tuned in order to produce the right output.
 - **CNNs** : Convolutional neural network. That is, a network which has at least one convolutional layer. A typical CNN also includes other types of layers, such as pooling layers and dense layers.
 - **Convolution** : The process of applying a kernel (filter) to an image
 - **Kernel / filter** : A matrix which is smaller than the input, used to transform the input into chunks
 - **Padding** : Adding pixels of some value, usually 0, around the input image
 - **Pooling**  The process of reducing the size of an image through downsampling.There are several types of pooling layers. For example, average pooling converts many values into a single value by taking the average. However, maxpooling is the most common.
 - **Maxpooling** : A pooling process in which many values are converted into a single value by taking the maximum value from among them.
 - **Stride** : the number of pixels to slide the kernel (filter) across the image.
 - **Downsampling** : The act of reducing the size of an image




