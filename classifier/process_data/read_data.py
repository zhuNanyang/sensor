import os
import numpy as np
import jieba

from jieba import analyse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import cv2
class WordCut(object):
    def __init__(self, stopword_path):
        self.stopwords_path = stopword_path
    def stopwordslist(self, filepath):
        stopwords = [line.strip() for line in open(filepath, "r", encoding="utf-8").readlines()]

        return stopwords
    def seg_sentence(self, sentence, stopwords_path=None):
        if stopwords_path is None:
            stopwords_path = self.stopwords_path
        sentence_seg = jieba.cut(sentence.strip(), cut_all=True, HMM=True)
        #print("sentence_seg:{}".format(" ".join(sentence_seg)))
        stopwords = self.stopwordslist(stopwords_path)
        content_clean = ""
        for word in sentence_seg:
            if word not in stopwords:
                if word != "\t":
                    content_clean += word
                    content_clean += " "
        return content_clean
class Read_data(object):

    def __init__(self, path, path_stopwords):
        super(Read_data, self).__init__()
        self.path = path
        self.path_stopwords = path_stopwords
    def read_data(self):
        Content = []
        Label = []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[0:10]:
                line_split = line.split("!")
                label = line_split[2]
                content = "".join(line_split[3:])
                Label.append(label)
                Content.append(content)
            return Content, Label

    def label_and_split_text(self):
        Content, Label = self.read_data()
        word_divider = WordCut(self.path_stopwords)
        Content_clean = []
        for content in Content:
            content = content.strip()
            content_clean = word_divider.seg_sentence(content)
            Content_clean.append(content_clean)
        print("Content_clean:{}".format(Content_clean))
        Label_ = []
        for label in Label:
            label = label.split("_")[2]
            Label_.append(label)
        print("Label:{}".format(Label_))
        return Content_clean, Label_
if __name__ == "__main__":
    path = os.path.join("H:/conpetition/competition_2/", "toutiao_cat_data.txt")
    path_stopwords = os.path.join("H:/conpetition/competition_2/", "stopwords.txt")
    Read_model = Read_data(path, path_stopwords)

    Content, Label = Read_model.label_and_split_text()
    print("Content.shape:{}".format(np.array(Content).shape))