import tensorflow as tf
import numpy as np

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
