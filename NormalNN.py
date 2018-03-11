import json as js
import numpy as np
import tensorflow as tf
import arrangeTrainQueen
import random,time


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
    # return tf.nn.leaky_relu(x)
    return tf.nn.sigmoid(x)


def fc_layer(inputtensor, size_in, size_out, name="fc"):
    with tf.variable_scope(name):
        w = tf.get_variable('W', initializer=tf.truncated_normal([size_in, size_out], stddev=0.1))
        b = tf.get_variable('B', initializer=tf.constant(0.1, shape=[size_out]))
        out = tf.matmul(inputtensor, w) + b
        act = activationfunc(out)
        reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
        # tf.summary.histogram("weights", w)
        # tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)
        return act, reg


def lptnnmodel(jsondatafilename, train_round):
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
    sortedpattern.sort(reverse=True)
    newdataDic = {}
    newdataDic['attr'] = np.empty((0, len(dataDic[sortedpattern[0]]['attribute'][0])))
    newdataDic['label'] = np.empty((0, len(dataDic[sortedpattern[0]]['label1'][0])))
    additionalfeature = 0
    ave = []
    summary = np.zeros(len(dataDic[sortedpattern[0]]['attribute'][0]))
    sample = 0
    for eachKey in sortedpattern:
        attrarray = np.array(dataDic[eachKey]['attribute'])
        labelarray = np.array(dataDic[eachKey]['label1'])
        summary += np.sum(attrarray, axis=0)
        # for prekey in newdataDic:
        #     containindex = stringcontain(prekey, eachKey)
        #     if containindex:
        #         newdataDic[prekey]['attr'] = np.vstack((newdataDic[prekey]['attr'], attrarray[:, containindex]))
        #         newdataDic[prekey]['label'] = np.vstack((newdataDic[prekey]['label'], labelarray))
        # print(1)
        attrarray = np.hstack((attrarray, np.ones((attrarray.shape[0], additionalfeature))*np.array(ave)))

        additionalfeature += 3
        sample += attrarray.shape[0]
        ave.reverse()
        for i in range(3):
            ave.append(summary[-(i+1)]/sample)
        ave.reverse()
        summary = summary[:-3]

        newdataDic['attr'] = np.vstack((newdataDic['attr'], attrarray))
        newdataDic['label'] = np.vstack((newdataDic['label'], labelarray))

    # print(newdataDic['attr'])

    ##shuffle samples
    sampleNumber = newdataDic['label'].shape[0]
    shuffleindex = [i for i in range(sampleNumber)]
    random.seed(1)
    random.shuffle(shuffleindex)
    newdataDic['attr'] = newdataDic['attr'][shuffleindex, :]
    newdataDic['label'] = newdataDic['label'][shuffleindex, :]
    trainPartSampleNum = int(0.8 * sampleNumber)
    newdataDic['traindata'] = {}
    newdataDic['traindata']['attr'] = newdataDic['attr'][:trainPartSampleNum, :]
    newdataDic['traindata']['label'] = newdataDic['label'][:trainPartSampleNum, :]
    newdataDic['testdata'] = {}
    newdataDic['testdata']['attr'] = newdataDic['attr'][trainPartSampleNum:, :]
    newdataDic['testdata']['label'] = newdataDic['label'][trainPartSampleNum:, :]

    ###above seperate origin data into training set and test set

    ##judge training round of every loss pattern, and arrange training turn of every pattern

    # with tf.variable_scope('share'):
    #     shareW1 = tf.get_variable('sW1', initializer=tf.truncated_normal(shareW1shape, stddev=0.1))
    #     shareb1 = tf.get_variable('B1', initializer=tf.constant(0.1, shape=[sharesizemid]))
    #     shareW2 = tf.get_variable('sW2', initializer=tf.truncated_normal(shareW2shape, stddev=0.1))
    #     shareb2 = tf.get_variable('B2', initializer=tf.constant(0.1, shape=[sharesizeout]))
    netName = 'nn'

    with tf.variable_scope(netName):
        inputData = tf.placeholder(tf.float32, name='input', shape=[None, sharesizein])
        midsize = int((2 * sharesizein) * 0.7)
        fc1, reg1 = fc_layer(inputData, sharesizein, midsize, name='fc1')
        fc2, reg2 = fc_layer(fc1, midsize, sharesizein, name='fc2')
        labels = tf.placeholder(tf.float32, name='labels', shape=[None, newdataDic['label'].shape[1]])

        shareW1 = tf.get_variable('sW1', initializer=tf.truncated_normal(shareW1shape, stddev=0.1))
        shareb1 = tf.get_variable('B1', initializer=tf.constant(0.1, shape=[sharesizemid]))
        shareW2 = tf.get_variable('sW2', initializer=tf.truncated_normal(shareW2shape, stddev=0.1))
        shareb2 = tf.get_variable('B2', initializer=tf.constant(0.1, shape=[sharesizeout]))

        hidden_layer = tf.matmul(fc2, shareW1) + shareb1
        hidden_layer_act = activationfunc(hidden_layer)
        out_layer = tf.matmul(hidden_layer_act, shareW2) + shareb2
        reg = tf.nn.l2_loss(shareW1) + tf.nn.l2_loss(shareW2) + tf.nn.l2_loss(shareb1) + tf.nn.l2_loss(shareb2) + reg1 + reg2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=labels)) + 1e-4 * reg
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)
        # add accuracy graph node
        correct_prediction = tf.equal(tf.argmax(out_layer, axis=1), tf.argmax(labels, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        nnResult = [loss, train_step, accuracy]

    starttime = time.time()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # sortedpattern.sort(reverse=True)
    # print(lossPatternTree)



    ######## according to training round to run sess to train nn
    for _i in range(train_round):
        [loss, train] = sess.run([nnResult[0], nnResult[1]],
                                 feed_dict={netName + '/input:0': newdataDic['traindata']['attr'],
                                            netName + '/labels:0': newdataDic['traindata']['label']})
        # print(loss)
        ##every 1k round output a test accuracy
        if _i%1000 == 0:
            trainaccuracy = sess.run(nnResult[2],
                                     feed_dict={netName + '/input:0': newdataDic['traindata']['attr'],
                                                netName + '/labels:0': newdataDic['traindata']['label']})

            testaccuracy = sess.run(nnResult[2],
                                    feed_dict={netName + '/input:0': newdataDic['testdata']['attr'],
                                               netName + '/labels:0': newdataDic['testdata']['label']})

            print('train:', trainaccuracy)
            print('test: ', testaccuracy)


    # break
    endtime = time.time()
    timecost = endtime - starttime
    print('time cost:', timecost)

    return


def main():
    lptnnmodel('posturedata.json', 10000)


if __name__ == '__main__':
    main()