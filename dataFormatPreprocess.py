import json as js

labelstr1 = 'label1'
labelstr2 = 'label2'
labelname = 'labelname'
attribute = 'attribute'
attributename = 'attributename'
with open('lossPatterns.txt', 'r') as rf:
    losspatternfname = rf.readlines()
    # print(losspatternfname)

    dataDic = {}
    for eachfname in losspatternfname:
        fname = eachfname.rstrip()
        with open(fname, 'r') as f:
            pattern = fname.split('.')[0]
            dataDic[pattern] = {}
            dataDic[pattern][labelstr1] = []
            dataDic[pattern][labelstr2] = []
            dataDic[pattern][attribute] = []
            dataDic[pattern][attributename] = []
            dataDic[pattern][labelname] = []
            attributenameline = f.readline()
            attributenamewords = attributenameline.rstrip().split(',')
            dataDic[pattern][labelname] = attributenamewords[:2]
            dataDic[pattern][attributename] = attributenamewords[2:]
            # print(dataDic[pattern][labelname], dataDic[pattern][attributename])

            for line in f:
                # print(line)
                attr = line.rstrip().split(',')

                ###label class
                label1 = [0 for _ in range(5)]
                label1[int(attr[0]) - 1] = 1
                dataDic[pattern][labelstr1].append(label1)

                ###label user
                label2 = [0 for _ in range(14)]
                if int(attr[1]) < 3:
                    label2[int(attr[1])] = 1
                else:
                    label2[int(attr[1]) - 1] = 1

                dataDic[pattern][labelstr2].append(label2)


                ###attributes
                dataDic[pattern][attribute].append(list(map(float, attr[2:])))
                # dataDic[pattern][attribute].append([float(x) for x in attr[2:]])

    with open('posturedata.json', 'w') as wf:
        js.dump(dataDic, wf)