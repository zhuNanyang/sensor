import os
import csv
import time

import datetime
import random
from collections import Counter

from math import sqrt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score ,precision_score, recall_score

from config import Config
from dataset import Dataset
from bilstm_attention import BiLSTMAttention

from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings, Batcher
def mean(item):
    return sum(item) / len(item)
def genMetrics(trueY, predY, binaryPredY):
    # 生成acc和auc的值

    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)

    recall = recall_score(trueY, binaryPredY)

    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)

def model(trainReviews, trainLabels, evalReviews, evalLabels):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = BiLSTMAttention()
            # 实例化BiLM对象，这个必须放置在全局下，不能在elmo函数中定义，否则会出现重复生成tensorflow节点

            with tf.variable_scope("bilm", reuse=True):
                bilm = BidirectionalLanguageModel(
                    config.optionFile,
                    config.weightFile,
                    use_character_inputs=False,
                    embedding_weight_file=config.tokenEmbeddingFile
                )
            inputData = tf.placeholder("int32", shape=[None, None])











            # 调用bilm中的__call__ 方法生成op对象
            inputEmbeddingOp = bilm(inputData)
            # 计算elmoInput向量表示
            elmoInput = weight_layers("input", inputEmbeddingOp, l2_coef=0.0)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            # 定义优化函数 传入学习速率参数
            optimizer = tf.train.AdamOptimizer(config.training.learningRate)
            # 计算梯度 得到梯度和变量
            gradsAndVars = optimizer.compute_gradients(cnn.loss)
            # 将梯度应用到变量下 生成训练器
            trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

            # 用summary绘制tensorBoard
            gradSummaries = []
            for g, v in gradsAndVars:
                if g is not None:
                    tf.summary.histogram("{}/grad/hist".format(v.name), g)

                    tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
            print("Writing to {}\n".format(outDir))
            lossSummary = tf.summary.scalar("loss", cnn.loss)
            summaryOp = tf.summary.merge_all()

            trainSummaryDir = os.path.join(outDir, "train")
            trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

            evalSummaryDir = os.path.join(outDir, "eval")
            evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)










            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # 保存模型的一种方式，保存为pb文件
            #         builder = tf.saved_model.builder.SavedModelBuilder("../model/textCNN/savedModel")
            sess.run(tf.global_variables_initializer())
            def elmo(reviews, inputData):
                """
                对每个输入的batcher都动态的生成词向量表示
                """
                # TokenBatcher是生成词表示的batch类
                batcher = TokenBatcher(config.vocabFile)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    # 生成batch数据
                    inputDataIndex = batcher.batch_sentences(reviews)
                    #print("inputDataIndex:{}".format(inputDataIndex))

                    # 计算ELMo的向量表示
                    elmoInputVec = sess.run([elmoInput["weighted_op"]],
                                            feed_dict={inputData: inputDataIndex})
                    return elmoInputVec

            def devStep(batchX, batchY):
                """
                验证函数
                """
                feed_dict = {
                    cnn.inputX: elmo(batchX, inputData)[0],
                    cnn.inputY: np.array(batchY, dtype="float32"),
                    cnn.dropoutKeepProb: 1.0
                }
                summary, step, loss, predictions, binaryPreds = sess.run(
                    [summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                    feed_dict)

                acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)

                evalSummaryWriter.add_summary(summary, step)

                return loss, acc, auc, precision, recall
            for i in range(config.training.epoches):
                print("start training model")
                for batchTrain in data.nextBatch(trainReviews, trainLabels, config.batchSize):
                    #print("batchTrain[0]:{}".format(batchTrain[0]))
                    #print("batchTrain[1]:{}".format(batchTrain[1]))
                    elmoInputVec = elmo(batchTrain[0], inputData)
                    feed_dict = {
                        cnn.inputX: elmoInputVec[0], # inputX 直接用动态生成的ELMo向量表示带入
                        cnn.inputY: np.array(batchTrain[1], dtype="float32"),
                        cnn.dropoutKeepProb: 0.5
                    }

                    _, summary, step, loss, predictions, binaryPreds = sess.run([trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds], feed_dict)

                    timeStr = datetime.datetime.now().isoformat()
                    acc, auc, precision, recall = genMetrics(batchTrain[1], predictions,binaryPreds)
                    print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step,
                                                                                                       loss, acc, auc,
                                                                                                       precision,
                                                                                                       recall))







                    trainSummaryWriter.add_summary(summary, step)
                    currentStep = tf.train.global_step(sess, globalStep)
                    if currentStep % config.training.evaluateEvery == 0:
                        print("\nEvaluation:")

                        losses = []
                        accs = []
                        aucs = []
                        precisions = []
                        recalls = []
                        for batchEval in data.nextBatch(evalReviews, evalLabels, config.batchSize):
                            loss, acc, auc, precision, recall = devStep(batchEval[0], batchEval[1])









                            losses.append(loss)
                            accs.append(acc)
                            aucs.append(auc)
                            precisions.append(precision)

                            recalls.append(recall)

                        time_str = datetime.datetime.now().isoformat()
                        print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str,
                                                                                                           currentStep,
                                                                                                           mean(losses),
                                                                                                           mean(accs),
                                                                                                           mean(aucs),
                                                                                                           mean(precisions),
                                                                                                           mean(recalls)))


if __name__ == "__main__":
    config = Config()
    data = Dataset()
    trainReviews = data.trainReviews
    trainLabels = data.trainLabels
    evalReviews = data.evalReviews
    evalLabels = data.evalLabels
    model(trainReviews, trainLabels, evalReviews, evalLabels)