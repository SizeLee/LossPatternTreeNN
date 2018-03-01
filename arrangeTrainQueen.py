def getTrainqueen(sortedPattern, sampleNumofPattern):
    sortedPattern.sort(reverse=True)
    trainqueen = choosequeenmodel(sortedPattern, sampleNumofPattern, 1)

    return trainqueen


def choosequeenmodel(sortedPattern, sampleNumofPattern, modeltype):
    if modeltype == 0:
        return patternAverageSorted_OneTraversal(sortedPattern, 1000)
    elif modeltype == 1:
        return patternAverageSorted_SeveralTraversals(sortedPattern, 1000, 10)
    elif modeltype == 2:
        return weighting_by_sample_num(sortedPattern, sampleNumofPattern, 100000)
    elif modeltype == 3:
        return weighting_by_sample_num_severaltraversals(sortedPattern, sampleNumofPattern, 10000, 10)


def patternAverageSorted_OneTraversal(sortedpattern, trainround):## here set traininground of every pattern
    result = []
    for eachpattern in sortedpattern:
        result.append((eachpattern, trainround))  ###eachpattern arrange 'trainround' times of trainround
    return result


def patternAverageSorted_SeveralTraversals(sortedpattern, trainround, traversals):
    return patternAverageSorted_OneTraversal(sortedpattern, trainround)*traversals


def weighting_by_sample_num(sortedPattern, sampleNumofPattern, totalround):
    totalnum = sum([sampleNumofPattern[key] for key in sortedPattern])
    patternweight = {}
    for eachkey in sampleNumofPattern:
        patternweight[eachkey] = 1/float(sampleNumofPattern[eachkey])

    totalweight = sum([patternweight[key] for key in sortedPattern])
    result = []
    for eachkey in sortedPattern:
        result.append((eachkey, int(totalround*patternweight[eachkey]/totalweight + 1)))

    return result


def weighting_by_sample_num_severaltraversals(sortedPattern, sampleNumofPattern, onetraversalround, traversals):
    return weighting_by_sample_num(sortedPattern, sampleNumofPattern, onetraversalround)*traversals

