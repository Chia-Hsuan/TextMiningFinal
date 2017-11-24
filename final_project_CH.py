import json
import pandas as pd
import numpy as np
import nltk
from nltk.util import ngrams
from nltk import word_tokenize, pos_tag, FreqDist
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import svm, cross_validation, preprocessing, ensemble, metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso

with open('MicroblogandHeadlineTestdata/Microblogs_Testdata_withscores.json', encoding='utf-8') as data_file:
	data = json.load(data_file)

df = pd.DataFrame.from_dict(data, orient='columns')
df['tokenized_sents'] = df['text'].str.split()
#print(df['text'])



# N-gram
def ngram(text):
	n_grams = ngrams(text.lower().split(), 3)
	return list(n_grams)
	
df['n-gram'] = df['text'].apply(ngram)
#print(df['n-gram'])


# Vader
def s_score(text):
	score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
	return score

df['vader_score'] = df['text'].apply(s_score)
#print(df['vader_score'])

X = np.array(df[['vader_score']])
y = np.array(df['sentiment score'])
train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y, test_size = 0.3)


# pos_tag
def pos_tagger(text, catagory):
	tag = pos_tag(text.lower().split())
	word_tag = FreqDist(tag)
	if catagory == 'adj':
		word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'JJ' or wt[1] == 'JJR' or wt[1] == 'JJS']
	elif catagory == 'adv':
		word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'RB' or wt[1] == 'RBR' or wt[1] == 'RBS']
	elif catagory == 'verb':
		word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'VB' or wt[1] == 'VBD' or wt[1] == 'VBG' or wt[1] == 'VBN' or wt[1] == 'VBP' or wt[1] == 'VBZ']
	return list(word)

df['pos-tag-adj'] = df['text'].apply(pos_tagger, catagory='adj')
df['pos-tag-adv'] = df['text'].apply(pos_tagger, catagory='adv')
df['pos-tag-verb'] = df['text'].apply(pos_tagger, catagory='verb')
print(df['pos-tag-adj'])
print(df['pos-tag-adv'])
print(df['pos-tag-verb'])


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
#print(accuracy)


# LASSO
lassoReg = Lasso(alpha=0.3, normalize=True)
lassoReg.fit(train_X,train_y)
pred = lassoReg.predict(test_X)
lasso_score = lassoReg.score(test_X,test_y)
#print(lasso_score)
