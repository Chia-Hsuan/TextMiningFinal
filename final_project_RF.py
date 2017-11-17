import json
import pandas as pd
import nltk
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import svm, cross_validation, preprocessing, ensemble, metrics, tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np

with open('MicroblogandHeadlineTestdata/Microblogs_Testdata_withscores.json', encoding='utf-8') as data_file:
	data = json.load(data_file)

df = pd.DataFrame.from_dict(data, orient='columns')
df['tokenized_sents'] = df['text'].str.split()
#print(df['text'])

def ngram(text):
	n_grams = ngrams(text.lower().split(), 3)
	return list(n_grams)
	
df['n-gram'] = df['text'].apply(ngram)
#print(df['n-gram'])


def s_score(text):
	score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
	return score

df['vader_score'] = df['text'].apply(s_score)
#print(df['vader_score'])



X = np.array(df[['vader_score']])
y = np.array(df['sentiment score'])
train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y, test_size = 0.3)

# Decision Tree
clf = tree.DecisionTreeClassifier()
sen_clf = clf.fit(X, y)
test_y_predicted = sen_clf.predict(test_X)
#print(test_y_predicted)


# Random Forest

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y, test_size = 0.3)
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(train_X, train_y)
test_y_predicted = forest.predict(test_X)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)




'''
X = np.array(df[['vader_score']])
y = np.array(df['sentiment score'])

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

print(X)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVR()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
'''
