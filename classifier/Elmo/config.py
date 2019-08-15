import os
class TrainingConfig(object):
    epoches = 10

    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

class ModelConfig(object):
    embeddingSize = 256 # 这个值和Elmo的词向量大小一致
    hiddenSizes = [128]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    sequenceLength = 200 # 取得了所有序列的长度的均值
    batchSize = 128
    dataSource = os.path.join("H:/conpetition/competition_2/Elmo/modelParams", "labeledTrain.csv")
    stopWordSource = os.path.join("H:/conpetition/competition_2/Elmo/modelParams", "english")
    optionFile = os.path.join("H:/conpetition/competition_2/Elmo/modelParams", "elmo_options.json")
    weightFile = os.path.join("H:/conpetition/competition_2/Elmo/modelParams", "elmo_weights.hdf5")

    vocabFile = os.path.join("H:/conpetition/competition_2/Elmo/modelParams", "vocab.txt")
    tokenEmbeddingFile = os.path.join("H:/conpetition/competition_2/Elmo/modelParams", "elmo_token_embeddings.hdf5")
    numClasses = 2
    rate = 0.8
    training =TrainingConfig()
    model = ModelConfig()