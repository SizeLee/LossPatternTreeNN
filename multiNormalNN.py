import json as js
import numpy as np
import tensorflow as tf
import arrangeTrainQueen
import random


def stringcontain(strcontained, str):
    containindex = []
    for i in range(len(strcontained)):
        if strcontained[i] == '1' and str[i] != '1':
            containindex = []
            break
        elif strcontained[i] == '1' and str[i] == '1':
            containindex.append(i)

    return containindex


def activationfunc(x):
    return tf.nn.leaky_relu(x)
    # return tf.nn.sigmoid(x)


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


def parseLossPatternAndBuildNN(losspattern, lptree, sharesizein, inputdatadim, labeldim, learning_rate):
    sharesizein = 36
    sharesizemid = 20
    sharesizeout = 5
    shareW1shape = [sharesizein, sharesizemid]
    shareW2shape = [sharesizemid, sharesizeout]

    featureNum = int(0)
    for i in losspattern:
        featureNum += int(i)

    with tf.variable_scope(losspattern):
        inputData = tf.placeholder(tf.float32, name='input', shape=[None, inputdatadim])
        midsize = int((featureNum + sharesizein) * 0.7)
        fc1 = fc_layer(inputData, featureNum, midsize, name='fc1')
        fc2 = fc_layer(fc1, midsize, sharesizein, name='fc2')
        labels = tf.placeholder(tf.float32, name='labels', shape=[None, labeldim])

        shareW1 = tf.get_variable('sW1', initializer=tf.truncated_normal(shareW1shape, stddev=0.1))
        shareb1 = tf.get_variable('B1', initializer=tf.constant(0.1, shape=[sharesizemid]))
        shareW2 = tf.get_variable('sW2', initializer=tf.truncated_normal(shareW2shape, stddev=0.1))
        shareb2 = tf.get_variable('B2', initializer=tf.constant(0.1, shape=[sharesizeout]))

        hidden_layer = tf.matmul(fc2, shareW1) + shareb1
        hidden_layer_act = activationfunc(hidden_layer)
        out_layer = tf.matmul(hidden_layer_act, shareW2) + shareb2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)
        # add accuracy graph node
        correct_prediction = tf.equal(tf.argmax(out_layer, axis=1), tf.argmax(labels, axis=1))
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


def lptnnmodel(jsondatafilename, competition_trainround, eachroundtimes):
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
        ##################it seems mix different losspattern's data is not a good choice,
        ################## maybe different losspattern's data is collected under different state of sensors.
        ################## so when mixing, training result is underfiting.
        ################# Here annotate code below about mixing.
        # for prekey in newdataDic:
        #     containindex = stringcontain(prekey, eachKey)
        #     if containindex:
        #         newdataDic[prekey]['attr'] = np.vstack((newdataDic[prekey]['attr'], attrarray[:, containindex]))
        #         newdataDic[prekey]['label'] = np.vstack((newdataDic[prekey]['label'], labelarray))
        # print(1)

        newdataDic[eachKey] = {}
        newdataDic[eachKey]['attr'] = attrarray
        newdataDic[eachKey]['label'] = labelarray

    ##shuffle samples
    for eachKey in newdataDic:
        sampleNumber = newdataDic[eachKey]['label'].shape[0]
        shuffleindex = [i for i in range(sampleNumber)]
        random.seed(1)
        random.shuffle(shuffleindex)
        newdataDic[eachKey]['attr'] = newdataDic[eachKey]['attr'][shuffleindex, :]
        newdataDic[eachKey]['label'] = newdataDic[eachKey]['label'][shuffleindex, :]
        trainPartSampleNum = int(0.8 * sampleNumber)
        newdataDic[eachKey]['traindata'] = {}
        newdataDic[eachKey]['traindata']['attr'] = newdataDic[eachKey]['attr'][:trainPartSampleNum, :]
        newdataDic[eachKey]['traindata']['label'] = newdataDic[eachKey]['label'][:trainPartSampleNum, :]
        newdataDic[eachKey]['testdata'] = {}
        newdataDic[eachKey]['testdata']['attr'] = newdataDic[eachKey]['attr'][trainPartSampleNum:, :]
        newdataDic[eachKey]['testdata']['label'] = newdataDic[eachKey]['label'][trainPartSampleNum:, :]

    ###above seperate origin data into training set and test set

    ##judge training round of every loss pattern, and arrange training turn of every pattern
    sampleNum = {}
    for eachKey in sortedpattern:
        sampleNum[eachKey] = newdataDic[eachKey]['label'].shape[0]
    trainQueen = arrangeTrainQueen.getTrainqueen(sortedpattern, sampleNum)

    # with tf.variable_scope('share'):
    #     shareW1 = tf.get_variable('sW1', initializer=tf.truncated_normal(shareW1shape, stddev=0.1))
    #     shareb1 = tf.get_variable('B1', initializer=tf.constant(0.1, shape=[sharesizemid]))
    #     shareW2 = tf.get_variable('sW2', initializer=tf.truncated_normal(shareW2shape, stddev=0.1))
    #     shareb2 = tf.get_variable('B2', initializer=tf.constant(0.1, shape=[sharesizeout]))

    lossPatternTree = {}
    for eachKey in dataDic:
        # print(eachKey)
        parseLossPatternAndBuildNN(eachKey, lossPatternTree, sharesizein,
                                   newdataDic[eachKey]['traindata']['attr'].shape[1],
                                   newdataDic[eachKey]['traindata']['label'].shape[1], learning_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # sortedpattern.sort(reverse=True)
    # print(lossPatternTree)

    for eachKey, trainround in trainQueen:
        # print(newdataDic[eachKey]['label'].shape)

        ######## according to training round to run sess to train nn
        for _ in range(trainround):
            [loss, train] = sess.run([lossPatternTree[eachKey][0], lossPatternTree[eachKey][1]],
                                     feed_dict={eachKey + '/input:0': newdataDic[eachKey]['traindata']['attr'],
                                                eachKey + '/labels:0': newdataDic[eachKey]['traindata']['label']})
            # print(loss)

        trainaccuracy = []
        testaccuracy = []
        for ackey in sortedpattern:
            accuracy = sess.run(lossPatternTree[ackey][2],
                                feed_dict={ackey + '/input:0': newdataDic[ackey]['traindata']['attr'],
                                           ackey + '/labels:0': newdataDic[ackey]['traindata']['label']})
            trainaccuracy.append(accuracy)

            accuracy = sess.run(lossPatternTree[ackey][2],
                                feed_dict={ackey + '/input:0': newdataDic[ackey]['testdata']['attr'],
                                           ackey + '/labels:0': newdataDic[ackey]['testdata']['label']})
            testaccuracy.append(accuracy)

        print('train:', trainaccuracy)
        print('test: ', testaccuracy)
        # break

    print('start competition')
    for _ in range(competition_trainround):
        worstkey = sortedpattern[trainaccuracy.index(min(trainaccuracy))]
        for _i in range(eachroundtimes):
            [loss, train] = sess.run([lossPatternTree[worstkey][0], lossPatternTree[worstkey][1]],
                                     feed_dict={worstkey + '/input:0': newdataDic[worstkey]['traindata']['attr'],
                                                worstkey + '/labels:0': newdataDic[worstkey]['traindata']['label']})

        trainaccuracy = []
        testaccuracy = []
        for ackey in sortedpattern:
            accuracy = sess.run(lossPatternTree[ackey][2],
                                feed_dict={ackey + '/input:0': newdataDic[ackey]['traindata']['attr'],
                                           ackey + '/labels:0': newdataDic[ackey]['traindata']['label']})
            trainaccuracy.append(accuracy)

            accuracy = sess.run(lossPatternTree[ackey][2],
                                feed_dict={ackey + '/input:0': newdataDic[ackey]['testdata']['attr'],
                                           ackey + '/labels:0': newdataDic[ackey]['testdata']['label']})
            testaccuracy.append(accuracy)

        print('train:', trainaccuracy)
        print('test: ', testaccuracy)
        # break

    trainaccuracyall = 0.
    testaccuracyall = 0.
    weightsum = 0.
    for i in range(len(sortedpattern)):
        weightsum += sampleNum[sortedpattern[i]]
        trainaccuracyall += trainaccuracy[i] * sampleNum[sortedpattern[i]]
        testaccuracyall += testaccuracy[i] * sampleNum[sortedpattern[i]]
    trainaccuracyall /= weightsum
    testaccuracyall /= weightsum
    print('train accuracy on all:', trainaccuracyall)
    print('test accuracy on all:', testaccuracyall)

    return


def main():
    lptnnmodel('posturedata.json', 50, 1000)


if __name__ == '__main__':
    main()