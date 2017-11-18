import json
import pandas as pd
import gensim
from sklearn import svm, cross_validation, preprocessing, decomposition
import numpy as np

# get data
with open('C:/data1.json', encoding='utf-8') as data_file:
    data = json.load(data_file)

df = pd.DataFrame.from_dict(data, orient='columns')

# preprocess
df['sentiment score'] = df['sentiment score'].apply(lambda x: (float(x)))
df['tokenized_sents'] = df['text'].str.split()

# build model
model = gensim.models.Word2Vec(df['tokenized_sents'], size=100, min_count=1)


def w2v(s):
    return model[s]


df['w2v'] = df['tokenized_sents'].apply(w2v)

df['w2v_sumAvg'] = df['w2v']

# w2v_sumAvg


def w2v_sumAvg():
    for x in range(0, df['w2v_sumAvg'].size):
        df['w2v_sumAvg'][x] = df['w2v'][x].mean(0)
        pass


w2v_sumAvg()


X = np.array(df['w2v_sumAvg'])
y = np.array(df['sentiment score'])

# split test and train data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


def _svm():
    clf = svm.SVR()
    clf.fit(X_train, y_train)

    def p(s):
        return clf.predict(s)
    df['p'] = df2['w2v_sumAvg'].apply(p)


# you need df['p'] as predicted value and df['sentiment score'] as gold standard
def final_score(df):
    a = (df['sentiment score'] * df['p']).sum()
    df['b'] = df['sentiment score'] ** 2
    df['c'] = df['p']**2

    b = (df['b'].sum())**(0.5)
    c = (df['c'].sum())**(0.5)

    cosine = a / (b * c)
    cosine_weight = df['p'].sum() / df['sentiment score'].sum()
    final_score = cosine * cosine_weight
    return final_score


_svm()

print(final_score(df))
