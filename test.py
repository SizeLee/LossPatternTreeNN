import tensorflow as tf
import numpy as np
import random

sess = tf.Session()

# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))
#
# node1 = tf.constant(3.0, dtype=tf.float32)
# node2 = tf.constant(4.0) # also tf.float32 implicitly
# # print(node1, node2)
# sess = tf.Session()
# # print(sess.run([node1, node2]))
# node3 = tf.add(node1, node2)
# print("node3:", node3)
# print("sess.run(node3):", sess.run(node3))
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b  # + provides a shortcut for tf.add(a, b)
# print(sess.run(adder_node, {a: 3, b: 4.5}))
# print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
#
# add_and_triple = adder_node * 3.
# print(sess.run(add_and_triple, {a: 3, b: 4.5}))
#
# W = tf.Variable([.3], dtype=tf.float32)
# b = tf.Variable([-.3], dtype=tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W*x + b
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(linear_model, {x: [[1, 2, 3, 4], [5, 6, 7, 8]]}))
#
# y = tf.placeholder(tf.float32)
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
#
# # fixW = tf.assign(W, [-1.])
# # fixb = tf.assign(b, [1.])
# # sess.run([fixW, fixb])
# # print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
#
# x2 = tf.placeholder(tf.float32)
# linear_model2 = W*x2 + b
#
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# sess.run(init) # reset values to incorrect defaults.
# for i in range(1000):
#   sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
#
# print(sess.run([W, b]))
# print(sess.run(linear_model2, {x2: [1, 2, 3, 4]}))
# print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

###############################


# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
# print(dataset1.output_types)  # ==> "tf.float32"
# print(dataset1.output_shapes)  # ==> "(10,)"
# # print(sess.run(tf.random_uniform([4])))
#
# dataset2 = tf.data.Dataset.from_tensor_slices(
#    (tf.random_uniform([4]),
#     tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
# print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
# print(dataset2.output_shapes)  # ==> "((), (100,))"
#
# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
# print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
# print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
#
# dataset = tf.data.Dataset.from_tensor_slices(
#    {"a": tf.random_uniform([4]),
#     "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
# print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
# print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
#
# iterator = dataset3.make_initializable_iterator()
# nextelement = iterator.get_next()
#
# sess.run(iterator.initializer)
# for i in range(4):
#     print(sess.run(nextelement))

#########################
# inc_dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([12, 16]))
# dec_dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([5, 7]))
# # dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
# batched_dataset = inc_dataset.batch(4)
#
# iterator = batched_dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# sess.run(iterator.initializer)
# print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
# print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
# print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])

######################################
###LOW LEVEL API###
######################################
# def func():
#     with tf.variable_scope(scope, reuse=True):
#         sw1 = tf.get_variable('sW1')
#     print(sess.run(sw1))
#
# with tf.variable_scope('myshare') as scope:
#     # shareW1 = tf.Variable(tf.truncated_normal([10, 10], stddev=0.1), name='sW1')
#     shareW1 = tf.get_variable('sW1', initializer=tf.truncated_normal([10, 10], stddev=0.1))
#
# sess.run(tf.global_variables_initializer())
# print(sess.run(shareW1))
# func()
#
# with tf.variable_scope("scope1"):
#     w1 = tf.get_variable("w1", shape=[])
#     w2 = tf.Variable(0.0, name="w2")
# with tf.variable_scope("scope1", reuse=True):
#     w1_p = tf.get_variable("w1", shape=[])
#     w2_p = tf.Variable(1.0, name="w2")
#
# print(w1 is w1_p, w2 is w2_p)

# def func():
#     sw1 = tf.get_variable('sW1')
#     print(sess.run(sw1))
#
#
# # shareW1 = tf.Variable(tf.truncated_normal([10, 10], stddev=0.1), name='sW1')
# shareW1 = tf.get_variable('sW1', initializer=tf.truncated_normal([10, 10], stddev=0.1))
#
# sess.run(tf.global_variables_initializer())
# print(sess.run(shareW1))
# func()

# a = tf.placeholder(tf.float32, name='testa')
# c = tf.shape(a)
# d = sess.run(c, feed_dict={'testa:0':[[1, 1, 1],[2, 2, 2]]})
# print(d[0])

# def nn():
#     inputData = tf.placeholder(tf.float32, name='testinput')
#     w = tf.get_variable('W', initializer=tf.constant(2.0, shape=[5, 4]))
#     b = tf.get_variable('B', initializer=tf.constant(0.1, shape=[4]))
#     print(w, b)
#     out = tf.matmul(inputData, w) + b
#     act = tf.nn.sigmoid(out)
#     return act
#
# with tf.variable_scope('1'):
#     a = nn()
#
# with tf.variable_scope('2'):
#     with tf.name_scope('1'):
#         w = tf.get_variable('W', initializer=tf.constant([[-1, -2, -3], [1, 2, -3], [1, 2, -3], [1, 2, 3]], dtype=tf.float32))
#         b = tf.matmul(a, w)
#         #print(w,b)
#         c = tf.nn.softmax_cross_entropy_with_logits(logits=b, labels=[[0, 1, 0], [0, 0, 1]])
#
# sess.run(tf.global_variables_initializer())
# print(sess.run((b, c), feed_dict={'1/testinput:0':np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])}))
#
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[2,3,4,5,6],[5,6,7,8,9]])
c = np.vstack((a,b[:, [0,2,4]]))
print(c)
ave = np.array([])
c = np.hstack((a,np.ones((2, 0)) * ave))
print(c)
ave = np.empty((0, 3))
a = np.ones((4,3))
v = np.vstack((ave, a))
print(v)
a = a[:, :-2]
print(a)

d = [1,2,3]
e = d.reverse()
print(d, e)
# tuplequeen = [(1,2), (2,3), (3,4), (4,5), (5,6), (5,6)]
# for a,b in tuplequeen:
#     print(a,b)
# print(tuplequeen*3)
# random.seed(1)
# a = [i for i in range(10)]
# random.shuffle(a)
# b = np.random.random_sample((10, 5))
# print(b)
# b = b[a, :]
# print(a,b)
# c = [1,2,6,3,8,4,5,3,2]
# i = c.index(max(c))
# print(i)