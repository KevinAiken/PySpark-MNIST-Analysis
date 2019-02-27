from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
import time


def parsePoint(line):
    values = line.split(',')
    values = [0 if e == '' else int(e) for e in values]
    return LabeledPoint(int(values[0]), values[1:])


sc = SparkContext(appName="MNISTDigitsDT")
fileNameTrain = 'mnist_train.csv'
fileNameTest = 'mnist_test.csv'
mnist_train = sc.textFile(fileNameTrain)
mnist_test = sc.textFile(fileNameTest)

labeledPoints = mnist_train.map(parsePoint)

(trainingData, testData) = labeledPoints.randomSplit([0.7, 0.3])


bestModel = None
bestTestErr = 100
maxDepths = range(4, 10)
maxTrees = range(3, 10)

for depthLevel in maxDepths:
    for treeLevel in maxTrees:
        start_time = time.time()
        model = RandomForest.trainClassifier(trainingData, numClasses=10, categoricalFeaturesInfo={},
                                             numTrees=treeLevel, featureSubsetStrategy="auto",
                                             impurity='gini', maxDepth=depthLevel, maxBins=32)
        predictions = model.predict(testData.map(lambda x: x.features))
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())

        print ('\maxDepth = {0:.1f}, trees = {1:.1f}: trainErr = {2:.5f}'.format(depthLevel, treeLevel, testErr))
        print("Prediction time --- %s seconds ---" % (time.time() - start_time))
        # print('Learned classification tree model:')
        # print(model.toDebugString())
        if (testErr < bestTestErr):
            bestModel = model
            bestTestErr = testErr
print ('Best Test Error: = {0:.3f}\n'.format(bestTestErr))
print bestModel

