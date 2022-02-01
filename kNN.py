from numpy import *
import operator

def createDataset():
    group = array ([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify(inX,dataSet,labels,k):
    """Function to classify input data into categories.
    A simple understanding of kNN Algorithm.
    Parameters:
    ------------
    inX: 2X1 list contating either 0's or 1's: Egs: [0,0],[0,1],[1,0],[1,1]
    dataSet: created dataset from the createDataset function.
    labels: labels /classes in the dataset.
    k: number of neighbors to be considered.

    Return:
    --------
    sortedClassCount[0][0] : class label to which the input inX belongs to.
    
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
         voterIlabel = labels[sortedDistIndices[i]]
         classCount[voterIlabel] = classCount.get(voterIlabel, 0) +1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
def file2matrix(filename):
    fr=open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index=0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector