from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
import os
from read_data import Read_data

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
import xgboost as xgb
import tqdm
import jieba
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import sequence ,text
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers import GlobalMaxPool1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout

from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
class Model(object):
    def __init__(self, train_con, test_con, train_label, test_label):

        self.train_con = train_con
        self.test_con = test_con
        self.train_label = train_label
        self.test_label = test_label
        self.Extract_feature = Extract_feature(self.train_con, self.test_con, self.train_label, self.test_label)
    def build_log_model(self):
        #train_data, test_data = self.Extract_feature.extract_count()
        train_data, test_data = self.Extract_feature.extract_tfidf()
        log_model = LogisticRegression(C = 1.0, penalty="l2")
        log_model.fit(train_data, self.train_label)
        predictions = log_model.predict(test_data)
        print("predictions:{}".format(predictions))
        classification = classification_report(y_true=self.test_label, y_pred=predictions)
        print("classification:{}".format(classification))
        f1 = f1_score(y_true=self.test_label, y_pred=predictions, average="micro")
        print("f1:{}".format(f1))
    def build_NB_model(self):

        train_data, test_data = self.Extract_feature.extract_tfidf()
        NB_clf = MultinomialNB()
        NB_clf.fit(train_data, self.train_label)
        prediction = NB_clf.predict(test_data)
        classification = classification_report(y_true=self.test_label, y_pred=prediction)
        print("NB_classification:{}".format(classification))
    def build_svm_model(self):

        train_data, test_data = self.Extract_feature.extract_tfidf()
        sc1 = StandardScaler(with_mean=False)
        sc1.fit(train_data)

        train_data = sc1.transform(train_data)
        test_data = sc1.transform(test_data)
        clf = SVC(C=1.0, probability=True)
        clf.fit(train_data, self.train_label)
        prediction = clf.predict(test_data)
        classification = classification_report(y_true=self.test_label, y_pred=prediction)
        print("SVM_classification:{}".format(classification))
    def build_xgboost_model(self):
        train_data, test_data = self.Extract_feature.extract_tfidf()

        #train_data, test_data, embedding_index = self.Extract_feature.extract_word2vec()
        clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                                subsample=0.8, nthread=10, learning_rate=0.1)

        clf.fit(train_data, self.train_label)
        prediction = clf.predict(test_data)
        classification = classification_report(y_true=self.test_label, y_pred=prediction)

        print("xgb_classification:{}".format(classification))
    def build_lstm_model(self):
        train_data, test_data, embedding_index = self.Extract_feature.extract_word2vec()
        train_label_binarize = np_utils.to_categorical(self.train_label)
        test_label_binarize = np_utils.to_categorical(self.test_label)
        token = text.Tokenizer(num_words=None)
        token.fit_on_texts(self.train_con + self.test_con)






        max_len = 30
        train_data = token.texts_to_sequences(self.train_con)
        test_data = token.texts_to_sequences(self.test_con)
        print("train_con:{}".format(len(train_data[0])))

        train_data_pad = sequence.pad_sequences(train_data, maxlen=max_len)
        test_data_pad = sequence.pad_sequences(test_data, maxlen=max_len)
        print("train_data_pad:{}".format(len(train_data_pad[0])))
        word_index = token.word_index
        print("word_index:{}".format(word_index))

        # 基于已有的数据集中的词汇创建一个词嵌入矩阵(Embedding Matrix)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            #print("embedding_vector:{}".format(embedding_vector))
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        model = Sequential()
        model.add(Embedding(len(word_index) + 1,
                  300,
                  weights=[embedding_matrix],
                  input_length=max_len,
                  trainable=False
                  ))
        model.add(SpatialDropout1D(0.3))
        model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.8))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.8))
        model.add(Dense(len(set(self.train_label))))

        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

        model.fit(train_data_pad, train_label_binarize, batch_size=64, epochs=300, verbose=1, validation_data=(test_data_pad, test_label_binarize), callbacks=[earlystop])


    def build_bidirection_lstm(self):
        train_data, test_data, embedding_index = self.Extract_feature.extract_word2vec()
        train_label_binarize = np_utils.to_categorical(self.train_label)
        test_label_binarize = np_utils.to_categorical(self.test_label)
        token = text.Tokenizer(num_words=None)
        token.fit_on_texts(self.train_con + self.test_con)

        max_len = 30
        train_data = token.texts_to_sequences(self.train_con)
        test_data = token.texts_to_sequences(self.test_con)
        print("train_data:{}".format(train_data))
        train_data_pad = sequence.pad_sequences(train_data, maxlen=max_len)

        test_data_pad = sequence.pad_sequences(test_data, maxlen=max_len)
        print("train_data_pad:{}".format(train_data_pad))
        word_index = token.word_index
        print("word_index:{}".format(word_index))
        """
        # 基于已有的数据集中的词汇创建一个词嵌入矩阵(Embedding Matrix)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            # print("embedding_vector:{}".format(embedding_vector))
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        model = Sequential()

        model.add(Embedding(len(word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False))

        model.add(SpatialDropout1D(0.3))
        model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.8))
        model.add(Dense(1024, activation="relu"))

        model.add(Dropout(0.8))
        model.add(Dense(len(set(self.train_label))))
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        model.fit(train_data_pad, train_label_binarize, batch_size=64, epochs=300, verbose=1,
                  validation_data=(test_data_pad, test_label_binarize), callbacks=[earlystop])

        """








    def build_GRU_model(self):
        train_data, test_data, embedding_index = self.Extract_feature.extract_word2vec()
        train_label_binarize = np_utils.to_categorical(self.train_label)
        test_label_binarize = np_utils.to_categorical(self.test_label)
        token = text.Tokenizer(num_words=None)
        token.fit_on_texts(self.train_con + self.test_con)
        max_len = 30
        train_data = token.texts_to_sequences(self.train_con)
        test_data = token.texts_to_sequences(self.test_con)
        print("train_data:{}".format(train_data))
        train_data_pad = sequence.pad_sequences(train_data, maxlen=max_len)
        test_data_pad = sequence.pad_sequences(test_data, maxlen=max_len)
        print("train_data_pad:{}".format(len(train_data_pad[0])))
        word_index = token.word_index
        #print("word_index:{}".format(word_index))
        # 基于已有的数据集中的词汇创建一个词嵌入矩阵(Embedding Matrix)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            # print("embedding_vector:{}".format(embedding_vector))
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        model = Sequential()
        model.add(Embedding(len(word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False
                            ))
        model.add(SpatialDropout1D(0.3))
        model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
        model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.8))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.8))
        model.add(Dense(len(set(self.train_label))))

        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

        model.fit(train_data_pad, train_label_binarize, batch_size=64, epochs=300, verbose=1,
                  validation_data=(test_data_pad, test_label_binarize), callbacks=[earlystop])
    def stacking(self):
        train_data, test_data = self.Extract_feature.extract_count()
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler, MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
        import xgboost as xgb
        from mlxtend.classifier import StackingClassifier

        import scipy as sc
        from sklearn import model_selection

        lasso = make_pipeline(SVC(C=2.1, gamma=0.005))
        rforest = make_pipeline(RandomForestClassifier(random_state=0, n_estimators=6))
        Gboost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,
                                            max_depth=12,
                                            max_features="sqrt",
                                            min_samples_leaf=15,min_samples_split=97,
                                             random_state=200)
        model_xgb = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=10,
                                      learning_rate=0.01, max_depth=11,
                                      n_estimators=500,
                                      reg_alpha=0.01, reg_lambda=5,
                                      subsample=0.5213,
                                      seed=1024, nthread=-1)


        lr = LogisticRegression()
        classifiers = [rforest, lasso, Gboost, model_xgb, lr]
        stregr = StackingClassifier(classifiers=classifiers, meta_classifier=lr)
        stregr.fit(train_data, self.train_label)

        prediction = stregr.predict(test_data)
        classification = classification_report(y_true=self.test_label, y_pred=prediction)
        print("classification:{}".format(classification))
        print("测试集的score:{}".format(stregr.score(test_data, self.test_label)))
        for clf, label in zip([rforest, lasso, Gboost, lr, model_xgb, stregr],
                              ['rf', 'svr', 'gboost', 'lr', 'xgb', 'stackingclassifier']):
            scores = model_selection.cross_val_score(clf, train_data, self.train_label, cv=3, scoring='accuracy')
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


def sent2vec(model, content):
    words = str(content).lower()
    word_vec = np.zeros((1, 300))
    for w in words:
        if w in model:
            word_vec += np.array([model[w]])


    return word_vec.mean(axis=0)
class Extract_feature(object):
    def __init__(self, train_con, test_con, train_label, test_label):
        self.train_con = train_con
        self.test_con = test_con
        self.train_label = train_label
        self.test_label = test_label
    def extract_tfidf(self):
        Tfid_vector = TfidfVectorizer(analyzer="word", max_features=None, min_df=3, max_df=0.5, use_idf=True, smooth_idf=True)
        Tfid_vector.fit(self.train_con + self.test_con)
        train_data =Tfid_vector.transform(self.train_con)
        test_data = Tfid_vector.transform(self.test_con)
        return train_data, test_data
    def extract_count(self):
        vectorizer = CountVectorizer(min_df=3, max_df=0.5, ngram_range=(1, 2))
        vectorizer.fit(self.train_con + self.test_con)
        train_data = vectorizer.transform(self.train_con)
        test_data = vectorizer.transform(self.test_con)

        return train_data, test_data

    def extract_word2vec(self):
        print("self.train_con:{}".format(self.train_con))
        content = [w.split(" ") for w in (self.train_con + self.test_con)]
        print("content:{}".format(content))
        model = Word2Vec(content, sg=0, size = 300, min_count=5, window=8, iter=10)
        model.init_sims(replace=True)
        print(model.wv.index2word)


        embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))
        print("embeddings_index:{}".format(embeddings_index))
        train_con = [w.split(" ") for w in (self.train_con)]
        test_con = [w.split(" ") for w in (self.test_con)]
        train_word2vec = [sent2vec(model, x) for x in train_con]
        test_word2vec = [sent2vec(model, x) for x in test_con]
        return np.array(train_word2vec), np.array(test_word2vec), embeddings_index











    def stacking(self):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler, MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
        import xgboost as xgb
        from mlxtend.classifier import StackingClassifier
        import scipy as sc
        lasso = make_pipeline(SVR(C=2.1, gamma=0.005))
        rforest = make_pipeline(RandomForestClassifier(random_state=0, n_estimators=6))
        Gboost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,
                                            max_depth=12,
                                            max_features="sqrt",
                                            min_samples_leaf=15,min_samples_split=97,
                                            loss='ls', random_state=200)
        model_xgb = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=10,
                                      learning_rate=0.01, max_depth=11)
if __name__ == "__main__":

    path = os.path.join("H:/conpetition/competition_2/", "toutiao_cat_data.txt")
    path_stopwords = os.path.join("H:/conpetition/competition_2/", "stopwords.txt")
    Data_model = Read_data(path, path_stopwords)
    Content, Label = Data_model.label_and_split_text()
    print("Content:{}".format(Content))
    Label_to_Int = LabelEncoder()
    label = Label_to_Int.fit_transform(Label)
    #print("label:{}".format(label))
    train_con, test_con, train_label, test_label = train_test_split(Content, label, train_size=0.9)
    print("train_con:{}".format(np.array(train_con).shape))
    #model = Model(train_con, test_con, train_label, test_label)
    #model.build_log_model()
    #model.build_NB_model()
    #model.build_svm_model()
    #model.build_xgboost_model()
    #model.build_lstm_model()
    #model.build_bidirection_lstm()
    #model.stacking()
    model = Extract_feature(train_con, test_con, train_label, test_label)
    model.extract_word2vec()