
with open('Postures.csv', 'r') as f:
    ##leave out first two lines
    for i in range(2):
        for line in f:
            break

    losspatternDic = {}

    for line in f:
        linetemp = line.strip()
        datas = linetemp.split(',')

        losspattern = ''
        # print(datas)
        for place in datas[2:]:
            if place != '?':
                losspattern += '1'
            else:
                losspattern += '0'

        # if len(losspattern) != 36:
        #     print('Error in data record')
        #     exit(1)
        if losspattern not in losspatternDic:
            losspatternDic[losspattern] = []
        losspatternDic[losspattern].append(line)
        # break

    # print(len(losspatternDic.keys()))

for eachkey in losspatternDic:
    print(eachkey, len(losspatternDic[eachkey]))
    filename = eachkey + '.csv'
    with open(filename, 'w') as wf:
        for eachline in losspatternDic[eachkey]:
            wf.write(eachline)