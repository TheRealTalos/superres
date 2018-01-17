import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.device("/gpu:0"):
	input_size = 16
	target_size = 32
	batch_size = 100

	input = tf.placeholder(tf.float32, [batch_size, input_size, input_size, 1])
	target = tf.placeholder(tf.float32, [batch_size, target_size**2])

	cn1 = 1000
	cn2 = 1000
	n1 = 5000

	conv1_w = tf.Variable(tf.random_normal([5, 5, 1, cn1], stddev=0.1))
	conv1_b = tf.Variable(tf.random_normal([cn1], stddev=0.1))

	conv2_w = tf.Variable(tf.random_normal([5, 5, cn1, cn2], stddev=0.1))
	conv2_b = tf.Variable(tf.random_normal([cn2], stddev=0.1))

	weights1 = tf.Variable(tf.random_normal([(input_size//4)*(input_size//4)*cn2, n1], stddev=0.1))
	biases1 = tf.Variable(tf.random_normal([n1], stddev=0.1))

	weights2 = tf.Variable(tf.random_normal([n1, target_size**2], stddev=0.1))
	biases2 = tf.Variable(tf.random_normal([target_size**2], stddev=0.1))

	conv1 = tf.nn.relu(tf.nn.conv2d(input, conv1_w, [1, 1, 1, 1], 'SAME') + conv1_b)
	pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

	conv2 = tf.nn.relu(tf.nn.conv2d(pool1, conv2_w, [1, 1, 1, 1], 'SAME') + conv2_b)
	pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
		              
	pool2 = tf.reshape(pool2, [batch_size, (input_size//4)*(input_size//4)*cn2])

	hidden = tf.nn.softmax(tf.matmul(pool2, weights1) + biases1)
	output = tf.matmul(hidden, weights2) + biases2

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))
	minimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	for e in range(10000):
	    input_batch, labels = mnist.train.next_batch(batch_size)
	    input_batch = tf.reshape(input_batch, [batch_size, 28, 28, 1])
	    target_batch = tf.image.resize_images(input_batch, [32, 32], method=1)
	    input_batch = tf.image.resize_images(input_batch, [16, 16], method=1)
	    input_batch = np.reshape(input_batch.eval(), input.shape)
	    target_batch = np.reshape(target_batch.eval(), target.shape)
	    
	    feed_dict={input: input_batch, target: target_batch}
	    
	    sess.run(minimizer, feed_dict=feed_dict)

	    if e % 100 is 0:
		train_accuracy = accuracy.eval(feed_dict=feed_dict)
		#print("Step {}, training batch accuracy {} %".format(e, train_accuracy*100))
		
		outimg = tf.reshape(output[0], [32, 32, 1])
		outimg = tf.image.convert_image_dtype(outimg, tf.uint8)
		outimg = tf.image.encode_png(outimg)
		
		file = open('xx/x{}.png'.format(e), 'wb+')
		file.write(sess.run(outimg, feed_dict=feed_dict))
		file.close()
