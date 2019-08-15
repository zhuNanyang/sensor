from config import Config
import pandas as pd
import random

class Dataset(Config):
    def __init__(self):
        self._dataSource = Config.dataSource

        self._stopWordSource = Config.stopWordSource
        self._optionFile = Config.optionFile
        self._weightFile = Config.weightFile
        self._vocabFile = Config.vocabFile
        self._tokenEmbeddingFile = Config.tokenEmbeddingFile

        self._sequenceLength = Config.sequenceLength
        self._embeddingSize = Config.model.embeddingSize
        self._batchSize = Config.batchSize
        self._rate = Config.rate
        self.trainReviews = []
        self.trainLabels = []
        self.evalReviews = []
        self.evalLabels = []
        self.dataGen()
    def _read_Data(self, filePath):
        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist()
        review = df["review"].tolist()
        reviews = [line.strip() for line in review]
        return reviews, labels
    def _fixedSeq(self, reviews):
        return [review[:self._sequenceLength] for review in reviews]
    def _genTrainEvalData(self, x, y, rate):

        y = [[item] for item in y]
        #print("y[:2]:{}".format(y[:2]))
        trainIndex = int(len(x) * rate)
        trainReviews = x[:trainIndex]
        trainLabels = y[:trainIndex]
        evalReviews = x[trainIndex:]
        evalLabels = y[trainIndex:]
        return trainReviews, trainLabels, evalReviews, evalLabels
    def dataGen(self):
        reviews, labels = self._read_Data(self._dataSource)
        #print("reviews[:2]:{}".format(reviews[:2]))

        #print("labels[0:2]:{}".format(labels[:2]))
        reviews = self._fixedSeq(reviews)
        #print("reviews[:2]:{}".format(reviews[:2]))

        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

    # 输出batch数据集
    def nextBatch(self, x, y, batchSize):
        """
        生成batch数据集，用生成器的方式输出
        """
        # 每一个epoch时，都要打乱数据集

        midVal = list(zip(x, y))

        random.shuffle(midVal)
        x, y = zip(*midVal)
        x = list(x)
        y = list(y)

        numBatches = len(x) // batchSize

        for i in range(numBatches):
            start = i * batchSize
            end = start + batchSize
            batchX = x[start: end]
            batchY = y[start: end]

            yield batchX, batchY
dataset = Dataset()

dataset.dataGen()

