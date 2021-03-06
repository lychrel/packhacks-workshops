# Convolutional Neural Networks
with Keras

---

## Why CNNs?
- Image inputs cause problems for fully-connected neural networks.

![Press Down Key](img/down-arrow.png)
+++
### Smaller image (CIFAR-10):

$32\times 32\times 3 = 3072$ parameters per neuron

(in the first layer)

### Larger image (most stuff):

$200\times 200\times 3 = 120,000$

parameters!

![Press Down Key](img/down-arrow.png)
+++
![gavin](img/gavin.jpg)
![Press Down Key](img/down-arrow.png)
+++
!["Are You Insane Man"](img/insane.gif)
![Press Down Key](img/down-arrow.png)
+++

- computationally expensive
- prone to overfitting

---

## CNN Inspiration:
### the Visual Cortex
- Regions of cells are sensitive to specific _regions_ of the visual field
- neurons in these regions only fire in presence of specific _features_: vertical, horizontal, or diagonal edges, etc.

---

### Convolutional Neural Networks
- assuming _image_ inputs (usually)
- neurons in a layer only connected to small _regions_ of previous layer, instead of fully-connected
- unique layer types
- operate on 3D _volumes_

![Press Down Key](img/down-arrow.png)

+++

### CNN as Layers
![conv1](img/CONV1.jpeg)

![Press Down Key](img/down-arrow.png)

+++

### [CNN as Volumes](http://cs231n.github.io/convolutional-networks/)
![conv2](img/CONV2.jpeg)

+++

## CNNs as Deep Learning

CNNs use convolutional and pooling layers to _extract features_ from a labeled dataset (features don't need to be provided)

---

# CNN LAYERS

![Press Down Key](img/down-arrow.png)
+++

## Convolutional Layers
- Slide (_concolve_) filters across image --> activation maps of image responses to filters
- Learn a set of filters appropriate to the problem

![Press Down Key](img/down-arrow.png)
+++
![Convolutions](img/filterGif.gif)
+++

## ReLu (Rectified Linear Units)
- Often follow convolutional layers
- Add nonlinearity to the net (e.g. changing all negative activations to 0)

![Press Down Key](img/down-arrow.png)
+++

## Pooling Layers
- "Downsampling the data"
- Reduce spatial size of representation to cut down computation and number of parameters (also controls overfitting)

![Press Down Key](img/down-arrow.png)
+++

## Dense (Fully-Connected) Layers
- Often at the end of the CNN
- Used for classification

---

## Example CNN Architecture
![convcar](img/convnet_coche.jpeg)

---

# MNIST Digit Classification
![Press Down Key](img/down-arrow.png)
+++

![mnist](img/5mnist.png)

---

# Implementation

![Press Down Key](img/down-arrow.png)
+++

### Keras
- high-level neural network API
- supports multiple backends (including TensorFlow)

### Tensorflow
- Google's (now open-source) Machine Learning framework

---

## Layers in Keras
![Press Down Key](img/down-arrow.png)
+++

### Input
```python
input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
```

![Press Down Key](img/down-arrow.png)
+++
### Convolutional (+ Activation)

```python
conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
```

![Press Down Key](img/down-arrow.png)
+++
### Pooling

```python
pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2, 2],
                                strides=2)
                                activation=tf.nn.relu)
```

![Press Down Key](img/down-arrow.png)
+++
### Dense + Dropout (Normalization)

```python
last_pool_flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])
                  dense = tf.layers.dense(inputs=pool2_flat,
                  units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense,
                            rate=0.4,
                            training=mode ==
                            tf.estimator.ModeKeys.TRAIN)
```

![Press Down Key](img/down-arrow.png)
+++
### Classification (SoftMax Cross-Entropy)

```python
# classes (digits 0 through 9)
logits = tf.layers.dense(inputs=dropout, units=10)

predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits,
                                   name="softmax_tensor")
}
```

---

# Training

- minimize a _loss function_ by adjusting filter values (weights) using _backpropagation_

---

### Backprop
*Forward Pass*: pass a training image ($24\times24\times3$) through network

- Network outputs a _prediction_ (class vector; initially bad)
- Calculate _loss_ between prediction and reality (label)

+++

Find $\frac{dL}{dW}$ (how loss changes with weights)

+++

*Backward Pass*: finding the weights that contributed _the most_ to the total loss and determining how to change them to decrease loss

+++

*Weight Update*: Update all filter weights in the _opposite direction_ of loss gradient $\frac{dL}{dW}$

---

## Important Learning Parameters
- *loss function*: how to quantify difference between prediction and reality
- *learning rate*: how much to update weights when traversing the loss gradient

![Press Down Key](img/down-arrow.png)
+++

- *epoch*: passing the entire dataset through the network once (generally split into batches)
- *batch*: a subset of the dataset fed into the neural network at once
- *iterations*: number of batches needed to complete an epoch

![Press Down Key](img/down-arrow.png)
+++

### Example:

For 2000 training images, split into _batches_ of 500, it will take 4 _iterations_ to complete one _epoch_.

---

## Hyperparameter Optimization
- _training_: batch size, number of epochs, learning rate, learning optimizer
- _layer_: number of filters, kernel size/shape, stride, padding, weight initializations, activation functions
- _architecture_: number of layers, dropout methods/values

---

## Hyperparameter Optimization Methods

Train with different sets of hyperparameters and see which combination yields the best results.


- Grid Search
- Random Search

![Press Down Key](img/down-arrow.png)
+++

### Grid vs Random Search
![search](img/searches.png)

---

### Putting it All Together

+++?code=create_model.py&lang=python

@[4](input layer)
@[7-12](first convolutional layer)
@[18-24](second convolutional layer)
@[24](second pooling layer)
@[27](flattening the pooling layer)
@[28](dense (fully connected) layer)
@[29](dropout normalization (40% chance of dropping any neuron in the dense layer))
@[33](logits (classes for prediction))
@[35-41](predicting the probability of each class using softmax)
@[47](calculate loss)
@[50-55](set learning parameters)
@[58-62](evaluate model accuracy)

---

## [Running it all: Google CoLab](https://tinyurl.com/packhacks-cnn)

[https://tinyurl.com/packhacks-cnn](https://tinyurl.com/packhacks-cnn)

---

# References

http://cs231n.github.io/convolutional-networks/
https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
https://seas.ucla.edu/~kao/nndl/lectures/cnn.pdf

---

# Additional Info

---

## One-Hot Encoding

- one-hot vectors are 0 in all dimensions but one (which is 1).
- avoids assuming order between categories, as integer encoding (1, 2, 3) does.
- e.x. [1,0,0], [0,1,0], [0,0,1] vs 0, 1, 2

---

## SoftMax (normalized exp(x))

- gives a list of values between 0 and 1 that sum to 1
  - a natural choice for probability
- exponentiates inputs then normalizes for pdf
- [more detailed exploration](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)

---

## Cross-Entropy

- measures how inefficient predictions are at describing the truth
- compares predicted probability distribution to true, one-hot encoded probability vector

+++
![Softmax (matrix form)](img/crossent.png)

for predicted distribution _y_, one-hot vector _y'_

---

## Backpropagation

- [detailed introduction](http://colah.github.io/posts/2015-08-Backprop/)

---

## Gradient Descent

- shifting variables small amounts in direction of reduced loss
- [mathy intro](https://en.wikipedia.org/wiki/Gradient_descent)

---

## MNIST Dimensionality

- [detailed exploration](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)

---

## Pooling Visualization

![pooling](img/pooling.png)

---

## CNN Hyperparameters

- depth, stride, and padding (determine conv layer output volume)
- filter sizes
- number of (conv) layers

---

## CNN additional detail

- [intro, pt 1](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/)
- [intro, pt 2](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

---

### Softmax Implementation

![Softmax](img/s1.png)

![Press Down Key](img/down-arrow.png)

+++

### equation form
![Softmax (equation form)](img/s1_eqns.png)

![Press Down Key](img/down-arrow.png)

+++

### matrix form
![Softmax (matrix form)](img/s1_matrices.png)

![Press Down Key](img/down-arrow.png)

+++

### defining the model

```python
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

---

## Convolutional Neural Networks (CNNs)
- assume image inputs
- 3D layers: width, height, depth
- neurons in a layer only connected to small _regions_ of previous layer, instead of fully-connected
- unique layer types

---

## Typical CNN Structure

1. convolutional modules (conv + pooling) for _feature extraction_
2. 1+ dense layer(s) for classification at end

---
