import json
import pandas as pd
import gensim
from sklearn import cross_validation, svm, ensemble, linear_model
from sklearn.svm import SVR
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# get data
file_location = 'C:/data1.json'
with open(file_location, encoding='utf-8') as data_file:
    data = json.load(data_file)
df = pd.DataFrame.from_dict(data, orient='columns')


# preprocess
df['sentiment score'] = df['sentiment score'].apply(lambda x: (float(x)))
df['tokenized_text'] = df['text'].apply(lambda x: CountVectorizer().build_analyzer()(x))
'''
tfidf = TfidfVectorizer()
temp = tfidf.fit_transform(df['text'])
temp.fi
'''
df_feature = pd.DataFrame()
# word2vec
word2vec = gensim.models.Word2Vec(df['tokenized_text'], size=3, min_count=1)
df['w2v'] = df['tokenized_text'].apply(lambda x: (word2vec[x]))
# word2vec___get mean
df_feature['w2v_mean'] = df['w2v'].apply(lambda x: (x.mean(0)))

# vader
df_feature['vader_compound'] = df['text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
df_feature['vader_compound'] = df_feature['vader_compound'] * 50

X = np.column_stack((np.array(df_feature['w2v_mean'].tolist()), np.array(df_feature['vader_compound'].tolist())))
y = np.array(df['sentiment score'].tolist())


# split test and train data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# regressors
clf = svm.SVR()
#clf = ensemble.GradientBoostingRegressor()
# clf = linear_model.LinearRegression()
# clf = linear_model.BayesianRidge()
clf.fit(X_train, y_train)
df_result = pd.DataFrame.from_dict(X_test, orient='columns')


# df_predict = df_result.apply(lambda x: (clf.predict([[-0.2], [0.4]])))
p = clf.predict(X_test)
g = y_test

# evaluate


def final_score(g, p):
    cosine = (g * p).sum() / ((g**2).sum()**0.5 * (p**2).sum()**0.5)
    cosine_weight = p.sum() / g.sum()
    return (cosine * cosine_weight)


print(final_score(g, p))
#print(clf.score(X_test, y_test))
