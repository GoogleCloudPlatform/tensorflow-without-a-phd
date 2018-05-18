## More about batch normalization
![Image](https://pbs.twimg.com/media/C2jZ9oEXEAEOuji.jpg)
You can find the theory of batch normalization explained here:

- [Video](https://www.youtube.com/watch?v=vq2nnJ4g6N0&t=76m43s)
- [Slides](https://docs.google.com/presentation/d/18MiZndRCOxB7g-TcCl2EZOElS5udVaCuxnGznLnmOlE/pub?slide=id.g1245051c73_0_25) - press s to open speaker notes for detailed explanations.

Tensorflow has both a low-level and a high-level implementation for batch normalization:

- [tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization)
- [tf.layers.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)

#### Low-level Tensorflow
The low-level tf.nn.batch_normalization function takes your inputs, subtracts the average and divides by the variance
that you pass in. It is up to you to compute both the batch statistics (average
and variance of neuron outputs across a batch) and their moving averages across multiple batches and use them apropriately at trainig and
test time. It is also up to you to compute your batch statistics
correctly depending on whether you are in a dense or a convolutional
layer. Sample code is available in [mnist_4.2_batchnorm_convolutional.py](mnist_4.2_batchnorm_convolutional.py)

#### High-level Tensorflow (tf.layers)

This version is is for models built using the tf.layers high-level API and wrapped in a tf.estimator.Estimator interface.

```Python
def model_fn(features, labels, mode):
    # ...  neural network layers ...
    logits = tf.layers.dense(Y4, 200, use_bias=False)
    bn = tf.layers.batch_normalization(logits,
        axis=1,
        center=True,
        scale=False,
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    Y5 = tf.nn.relu(bn)
    # ...  more neural network layers ...
```
A complete sample is available in [mlengine/trainer/task.py](/mlengine/trainer/task.py)

#### axis
The default is axis=-1 which means "last axis". This will work for both dense
and convolutional layers if they are organised as [batch, features]
or [batch, x, y, filters] for dense and convolutional layers respectively.

For dense layers, where the output looks like [batch, features], the correct value is axis=1.
For convolutional layers, where the output looks like [batch, x, y, filters]
it is axis=3. Batch norm collects and uses, for each neuron, statistics on the
output from that neuron across a batch. In a dense layer, one neuron has one
output per data item in the batch. In a convolutional layer, one neuron has
one output per data item in the batch and per x,y location. The axis parameter
is what identifies individual neurons, all other dimensions of your outputs
are for possible output values for that neuron. Using axis=1, stats will be
collected with tf.nn.moments([batch]). Using axis=3, stats will be collected
using tf.nn.moments([batch, x, y]) which are the correct populations for dense
and conv layers respectively.

#### center, scale
- a bias is not useful when using batch norm. Remove biases from layers regularized with batch norm.
- batch norm offset should always be used (replaces bias) 
- batch norm scale should be used with scale-dependant activation functions (sigmoid: yes, relu: no)

#### training
Pass in (mode == tf.estimator.ModeKeys.TRAIN) and batch norm will correctly accumulate batch statistics during training
and use them during evaluation and inference.

#### exponential moving averages of batch stats (mean and variance)
tf.layers.batch_normalization creates variables for the batch norm stats that need to be gathered. These variables are added to tf.GraphKeys.UPDATE_OPS and these UPDATE_OPS are ran automatically
when you use the tf.contrib.training.create_train_op function to compute the training_op that the Estimator API requires.
If you want to use a tf.train.XXXOptimizer directly, add a graph dependency on UPDATE_OPS so that updates happen before your train_op:
```Python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
# Note: This code is already present in the
# tf.contrib.training.create_train_op helper function.
# No need to duplicate if you are using that.
```

#### batch norm and activation functions
Batch norm is normally applied to the output of a neural network layer, before the activation function.
In the present tf.layers API (TF1.3), there is no one-line syntax for a dense layer with batch norm and relu. The layers API only offers an activity_regularizer parameter which is applied after the activation function.
Use layers without an activation function, then apply batch norm, then the activation.