import json as js
import numpy as np
import tensorflow as tf

def stringcontain(strcontained, str):
    containindex = []
    for i in range(len(strcontained)):
        if strcontained[i] == '1' and str[i] != '1':
            containindex = []
            break
        elif strcontained[i] == '1' and str[i] =='1':
            containindex.append(i)

    return containindex

def activationfunc(x):
    return tf.nn.sigmoid(x)

def fc_layer(inputtensor, size_in, size_out, name="fc"):
  with tf.variable_scope(name):
    w = tf.get_variable('W', initializer=tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.get_variable('B', initializer=tf.constant(0.1, shape=[size_out]))
    out = tf.matmul(inputtensor, w) + b
    act = activationfunc(out)
    # tf.summary.histogram("weights", w)
    # tf.summary.histogram("biases", b)
    # tf.summary.histogram("activations", act)
    return act

def parseLossPatternAndBuildNN(losspattern, lptree, sharesizein, label, learning_rate):
    featureNum = int(0)
    for i in losspattern:
        featureNum += int(i)

    with tf.variable_scope(losspattern):
        inputData = tf.placeholder(tf.float32, name='input')
        midsize = int((featureNum + sharesizein) * 0.7)
        fc1 = fc_layer(inputData, featureNum, midsize, name='fc1')
        fc2 = fc_layer(fc1, midsize, sharesizein, name='fc2')

    with tf.variable_scope('share', reuse=True):
        shareW1 = tf.get_variable('sW1')
        shareb1 = tf.get_variable('B1')
        shareW2 = tf.get_variable('sW2')
        shareb2 = tf.get_variable('B2')

    with tf.variable_scope(losspattern):
        hidden_layer = tf.matmul(fc2, shareW1) + shareb1
        hidden_layer_act = activationfunc(hidden_layer)
        out_layer = tf.matmul(hidden_layer_act, shareW2) + shareb2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=label))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)
        #todo add accuracy graph node
        correct_prediction = tf.equal(tf.argmax(out_layer, axis=1), tf.argmax(label, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        lptree[losspattern] = [loss, train_step, accuracy]
        # print(hidden_layer)
        # print(hidden_layer_act)
        # print(out_layer)
        # print(loss)
        # print(optimizer)
        # print(train_step)
        # print()

    return

def lptnnmodel(jsondatafilename):
    sharesizein = 36
    sharesizemid = 20
    sharesizeout = 5
    shareW1shape = [sharesizein, sharesizemid]
    shareW2shape = [sharesizemid, sharesizeout]
    learning_rate = 0.001

    with open(jsondatafilename, 'r') as f:
        dataDic = js.load(f)


    ####根据losspattern排序，将数据整合多次分配，根据各个pattern数据量来决定训练次数。
    # print(sortedpattern)
    sortedpattern = [eachkey for eachkey in dataDic]
    sortedpattern.sort()
    newdataDic = {}
    for eachKey in sortedpattern:
        attrarray = np.array(dataDic[eachKey]['attribute'])
        labelarray = np.array(dataDic[eachKey]['label1'])
        for prekey in newdataDic:
            containindex = stringcontain(prekey, eachKey)
            if containindex:
                newdataDic[prekey]['attr'] = np.vstack((newdataDic[prekey]['attr'], attrarray[:, containindex]))
                newdataDic[prekey]['label'] = np.vstack((newdataDic[prekey]['label'], labelarray))

        newdataDic[eachKey] = {}
        newdataDic[eachKey]['attr'] = attrarray
        newdataDic[eachKey]['label'] = labelarray

    ##todo judge training round of every loss pattern

    with tf.variable_scope('share'):
        shareW1 = tf.get_variable('sW1', initializer=tf.truncated_normal(shareW1shape, stddev=0.1))
        shareb1 = tf.get_variable('B1', initializer=tf.constant(0.1, shape=[sharesizemid]))
        shareW2 = tf.get_variable('sW2', initializer=tf.truncated_normal(shareW2shape, stddev=0.1))
        shareb2 = tf.get_variable('B2', initializer=tf.constant(0.1, shape=[sharesizeout]))

    lossPatternTree = {}
    for eachKey in dataDic:
        # print(eachKey)
        parseLossPatternAndBuildNN(eachKey, lossPatternTree, sharesizein, newdataDic[eachKey]['label'], learning_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sortedpattern.sort(reverse=True)
    # print(lossPatternTree)
    i = 0
    for eachKey in sortedpattern:
        # print(newdataDic[eachKey]['label'].shape)
        #todo according to training round to run sess to train nn
        for _ in range(1000):
            [loss, train] = sess.run([lossPatternTree[eachKey][0], lossPatternTree[eachKey][1]],
                                    feed_dict={eachKey+'/input:0':newdataDic[eachKey]['attr']})
            # print(loss)
        accuracy = sess.run(lossPatternTree[eachKey][2],
                            feed_dict={eachKey+'/input:0':newdataDic[eachKey]['attr']})
        print(accuracy)
        i += 1
        if i>3:
            break

    return


def main():
    lptnnmodel('posturedata.json')

if __name__ == '__main__':
    main()