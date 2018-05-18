import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

sess = tf.Session()
# x_vals = np.array([1., 3., 5., 7., 9.])
# x_data = tf.placeholder(tf.float32)
# m_const = tf.constant(3.)
# my_product = tf.multiply(x_data, m_const)
# for x_val in x_vals:
#     print(sess.run(my_product, feed_dict={x_data: x_val}))
#
# my_array = np.array([[1., 3., 5., 7., 9.],
#                      [-2., 0., 2., 4., 6.],
#                      [-6., -3., 0., 3., 6.]])
# x_vals = np.array([my_array, my_array + 1])
# x_data = tf.placeholder(tf.float32, shape=(3, 5))
# m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
# m2 = tf.constant([[2.]])
# a1 = tf.constant([[10.]])
# prod1 = tf.matmul(x_data, m1)
# prod2 = tf.matmul(prod1, m2)
# add1 = tf.add(prod2, a1)
# for x_val in x_vals:
#     print(sess.run(add1, feed_dict={x_data: x_val}))
#
# x_shape = [1, 4, 4, 1]
# x_val = np.random.uniform(size=x_shape)
# x_data = tf.placeholder(tf.float32, shape=x_shape)
# my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
# my_strides = [1, 2, 2, 1]
# mov_avg_layer= tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')
# def custom_layer(input_matrix):
#     input_matrix_sqeezed = tf.squeeze(input_matrix)
#     A = tf.constant([[1., 2.], [-1., 3.]])
#     b = tf.constant(1., shape=[2, 2])
#     temp1 = tf.matmul(A, input_matrix_sqeezed)
#     temp = tf.add(temp1, b) # Ax + b
#     return(tf.sigmoid(temp))
# with tf.name_scope('Custom_Layer') as scope:
#     custom_layer1 = custom_layer(mov_avg_layer)
# print(sess.run(custom_layer1, feed_dict={x_data: x_val}))
#
# x_vals = tf.linspace(-1., 1., 500)
# target = tf.constant(0.)
# l2_y_vals = tf.square(target - x_vals)
# l2_y_out = sess.run(l2_y_vals)
# l1_y_vals = tf.abs(target - x_vals)
# l1_y_out = sess.run(l1_y_vals)
# delta1 = tf.constant(0.25)
# phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
# phuber1_y_out = sess.run(phuber1_y_vals)
# delta2 = tf.constant(5.)
# phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
# phuber2_y_out = sess.run(phuber2_y_vals)
# x_vals = tf.linspace(-3., 5., 500)
# target = tf.constant(1.)
# targets = tf.fill([500,], 1.)
# hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
# hinge_y_out = sess.run(hinge_y_vals)
# xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
# xentropy_y_out = sess.run(xentropy_y_vals)
# xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_vals, labels=targets)
# xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)
# weight = tf.constant(0.5)
# xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
# xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)
# unscaled_logits = tf.constant([[1., -3., 10.]])
# target_dist = tf.constant([[0.1, 0.02, 0.88]])
# softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=unscaled_logits, labels=target_dist)
# print(sess.run(softmax_xentropy))
#
# unscaled_logits = tf.constant([[1., -3., 10.]])
# sparse_target_dist = tf.constant([2])
# sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=sparse_target_dist)
# print(sess.run(sparse_xentropy))
#
# x_array = sess.run(x_vals)
# plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
# plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
# plt.plot(x_array, phuber1_y_out, 'k-.', label='P-Huber Loss (0.25)')
# plt.plot(x_array, phuber2_y_out, 'g:', label='P-Huber Loss (5.0)')
# plt.ylim(-0.2, 0.4)
# plt.legend(loc='lower right', prop={'size': 11})
# plt.show()
#
# x_array = sess.run(x_vals)
# plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
# plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
# plt.plot(x_array, xentropy_sigmoid_y_out, 'k-.', label='Cross Entropy Sigmoid Loss')
# plt.plot(x_array, xentropy_weighted_y_out, 'g:', label='Weighted Cross Enropy Loss (x0.5)')
# plt.ylim(-1.5, 3)
# plt.legend(loc='lower right', prop={'size': 11})
# plt.show()

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))
my_output = tf.multiply(x_data, A)
loss = tf.square(my_output - y_target)
init = tf.initialize_all_variables()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

ops.reset_default_graph()
sess = tf.Session()
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
my_output = tf.add(x_data, A)
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)
init = tf.initialize_all_variables()
sess.run(init)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits( logits=my_output_expanded, labels=y_target_expanded)
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))

ops.reset_default_graph()
sess = tf.Session()
batch_size = 20
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
my_output = tf.multiply(x_data, A)
loss = tf.reduce_mean(tf.square(my_output - y_target))
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 5 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)