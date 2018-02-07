
with open('Postures.csv', 'r') as f:
    ###take first line
    line = f.readline()
    featurenames = line.strip()
    featurename = featurenames.split(',')

    ##leave out second two lines
    f.readline()

    losspatternDic = {}

    for line in f:
        linetemp = line.strip()
        datas = linetemp.split(',')

        losspattern = ''
        realline = ''
        # print(datas)
        for place in datas[2:]:
            if place != '?':
                losspattern += '1'
                realline += place
                realline += ','
            else:
                losspattern += '0'

        realline = datas[0] +',' + datas[1] + ',' + realline

        realline = realline.rstrip(',')
        realline += '\n'

        # if len(losspattern) != 36:
        #     print('Error in data record')
        #     exit(1)
        if losspattern not in losspatternDic:
            losspatternDic[losspattern] = []
        losspatternDic[losspattern].append(realline)
        # print(realline)
        # break

    # print(len(losspatternDic.keys()))

sortedfilename = []
for eachkey in losspatternDic:
    print(eachkey, len(losspatternDic[eachkey]))
    filename = eachkey + '.csv'
    sortedfilename.append(filename)

    featureExist = featurename[0] + ',' + featurename[1] + ','
    for i in range(len(eachkey)):
        if eachkey[i] == '1':
            featureExist += (featurename[i+2] + ',')

    featureExist = featureExist.rstrip(',')
    featureExist += '\n'

    with open(filename, 'w') as wf:
        wf.write(featureExist)
        for eachline in losspatternDic[eachkey]:
            wf.write(eachline)

sortedfilename.sort()
with open('lossPatterns.txt', 'w') as wl:
    for eachfilename in sortedfilename:
        wl.write(eachfilename + '\n')

