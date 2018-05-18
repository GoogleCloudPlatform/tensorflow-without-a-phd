![Image](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/img/93d5f08a4f82d4c.png)

This is support code for the codelab "[Tensorflow and deep learning - without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist)"

The presentation explaining the underlying concepts is [here](https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/pub).

The lab takes 2.5 hours and takes you through the design and optimisation of a neural network for recognising handwritten digits, from the simplest possible solution all the way to a recognition accuracy above 99%. It covers dense and convolutional networks, as well as techniques such as learning rate decay and dropout.

Installation instructions [here](INSTALL.txt). The short version is: install Python3, then pip3 install tensorflow and matplotlib.
   
The most advanced advanced neural network in this repo achieves 99.5% accuracy on the MNIST dataset (world best is 99.7%) and uses [batch normalization](README_BATCHNORM.md).

This lab uses low-level Tensorflow because it is intended as a starting point for
developers learning neural network techniques. It is important to see what is going on with
trainable variables (weights and biases) before moving to higher-level APIs that hide these concepts.
If you are looking for a high-level Tensorflow sample using layers, Estimator and Dataset APIs, you
will find it in the [mlengine](mlengine) folder. 

---

*Disclaimer: This is not an official Google product but sample code provided for an educational purpose*