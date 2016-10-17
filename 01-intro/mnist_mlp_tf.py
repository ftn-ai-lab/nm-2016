from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('/tmp/data/', one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.1
training_epochs = 15
batch_size = 100

# Network Parameters
n_hidden = 256  # number of features in hidden layer
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h']), biases['b'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'b': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# cross-entropy loss (log-loss)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        train_cost, train_acc = 0., 0.
        total_batch = int(data.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = data.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, batch_cost, batch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x,
                                                                                        y: batch_y})
            # Compute average loss
            train_cost += batch_cost / total_batch
            train_acc += batch_acc / total_batch
        # Display logs per epoch step
        test_cost, test_acc = sess.run([cost, accuracy], feed_dict={x: data.test.images,
                                                                    y: data.test.labels})
        print('Epoch: {}, train_cost={:.4f}, train_acc={:.4f}, test_cost={:.4f}, test_acc={:.4f}'
              .format(epoch + 1, train_cost, train_acc, test_cost, test_acc))
        
    print('Optimization Finished!')

