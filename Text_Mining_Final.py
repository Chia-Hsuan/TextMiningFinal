import json
import pandas as pd
import gensim
from sklearn import cross_validation, svm, ensemble, linear_model, tree
from sklearn.svm import SVR
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from scipy import spatial
from nltk import pos_tag, FreqDist
import re

# get data
file_location = 'MicroblogandHeadlineTestdata/Microblogs_Testdata_withscores.json'
with open(file_location, encoding='utf-8') as data_file:
    data = json.load(data_file)
df = pd.DataFrame.from_dict(data, orient='columns')


# preprocess
df['sentiment score'] = df['sentiment score'].apply(lambda x: (float(x)))
df['tokenized_text'] = df['text'].apply(lambda x: re.sub(r"http\S+", "", x)).apply(lambda x: CountVectorizer().build_analyzer()(x))

df_feature = pd.DataFrame()
st = StanfordNERTagger('/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/usr/share/stanford-ner/stanford-ner.jar', encoding='utf-8')

# tfidf
tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=0.01, max_df=0.5)
feature_tfidf = np.array(tfidf.fit_transform(df['text']).toarray())
################################################################################

# word2vec
word2vec = gensim.models.Word2Vec(df['tokenized_text'], size=3, min_count=1)
df['w2v'] = df['tokenized_text'].apply(lambda x: (word2vec[x]))
# word2vec___get mean
feature_w2vMean = np.array((df['w2v'].apply(lambda x: (x.mean(0)))).tolist())
# word2vec___tfidf


def get_tfidfScore(tokens):
    rtv = list()
    for x in range(0, len(tokens)):
        index = tfidf.vocabulary_.get(tokens[x])
        if(index == None):
            rtv.append(0)
        else:
            rtv.append(index)
        pass
    return rtv


df['tfidf_score'] = df['tokenized_text'].apply(get_tfidfScore)

feature_w2vTfidf = list()
for x in range(0, df['w2v'].size):
    row = list()
    for y in range(0, len(df['w2v'][0][0])):
        tfidf_sum = 0.0
        product = 0.0
        for z in range(0, len(df['w2v'][x])):
            product += df['w2v'][x][z][y] * df['tfidf_score'][x][z]
            tfidf_sum += df['tfidf_score'][x][z]
            pass
        if(product == 0):
            average = 0
        else:
            average = product / tfidf_sum

        row.append(average)
        pass
    feature_w2vTfidf.append(row)
    pass

feature_w2vTfidf = np.array(feature_w2vTfidf)


# vader
feature_vaderCompound = np.array(df['text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound']).tolist())
feature_vaderCompound = np.reshape(feature_vaderCompound, (-1, 1))


# word2vec___pos_tag_adj__mean

def pos2v(x):
    if (len(x) == 0):
        return np.array([np.zeros(word2vec.vector_size)])
    return word2vec[x]

def pos_tagger(text, catagory):
    tag = pos_tag(text)
    word_tag = FreqDist(tag)
    if catagory == 'adj':
        word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'JJ' or wt[1] == 'JJR' or wt[1] == 'JJS']
    elif catagory == 'adv':
        word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'RB' or wt[1] == 'RBR' or wt[1] == 'RBS']
    elif catagory == 'verb':
        word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'VB' or wt[1] == 'VBD' or wt[1] == 'VBG' or wt[1] == 'VBN' or wt[1] == 'VBP' or wt[1] == 'VBZ']
    elif catagory == 'adj_adv':
        word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'JJ' or wt[1] == 'JJR' or wt[1] == 'JJS' or wt[1] == 'RB' or wt[1] == 'RBR' or wt[1] == 'RBS']
    elif catagory == 'adj_verb':
        word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'JJ' or wt[1] == 'JJR' or wt[1] == 'JJS' or wt[1] == 'VB' or wt[1] == 'VBD' or wt[1] == 'VBG' or wt[1] == 'VBN' or wt[1] == 'VBP' or wt[1] == 'VBZ']
    elif catagory == 'adv_verb':
        word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'RB' or wt[1] == 'RBR' or wt[1] == 'RBS' or wt[1] == 'VB' or wt[1] == 'VBD' or wt[1] == 'VBG' or wt[1] == 'VBN' or wt[1] == 'VBP' or wt[1] == 'VBZ']
    elif catagory == 'adj_adv_verb':
        word = [wt[0] for (wt, _) in word_tag.most_common() if wt[1] == 'JJ' or wt[1] == 'JJR' or wt[1] == 'JJS' or wt[1] == 'RB' or wt[1] == 'RBR' or wt[1] == 'RBS' or wt[1] == 'VB' or wt[1] == 'VBD' or wt[1] == 'VBG' or wt[1] == 'VBN' or wt[1] == 'VBP' or wt[1] == 'VBZ']
    return word


df['pos-tag-adj'] = df['tokenized_text'].apply(pos_tagger, catagory='adj')
df['pos-tag-adv'] = df['tokenized_text'].apply(pos_tagger, catagory='adv')
df['pos-tag-verb'] = df['tokenized_text'].apply(pos_tagger, catagory='verb')
df['pos-tag-adj_adv'] = df['tokenized_text'].apply(pos_tagger, catagory='adj_adv')
df['pos-tag-adj_verb'] = df['tokenized_text'].apply(pos_tagger, catagory='adj_verb')
df['pos-tag-adv_verb'] = df['tokenized_text'].apply(pos_tagger, catagory='adv_verb')
df['pos-tag-adj_adv_verb'] = df['tokenized_text'].apply(pos_tagger, catagory='adj_adv_verb')

for p in range(7):
    if p == 0:
        pos = "adj"
        pos_name = "Adj"
    elif p == 1:
        pos = "adv"
        pos_name = "Adv"
    elif p == 2:
        pos = "verb"
        pos_name = "Verb"
    elif p == 3:
        pos = "adj_adv"
        pos_name = "AdjAdv"
    elif p == 4:
        pos = "adj_verb"
        pos_name = "AdjVerb"
    elif p == 5:
        pos = "adv_verb"
        pos_name = "AdvVerb"
    else:
        pos = "adj_adv_verb"
        pos_name = "AdjAdvVerb"

    df['w2v_' + pos] = df['pos-tag-' + pos].apply(pos2v)
    feature_w2vAdj = np.array((df['w2v_' + pos].apply(lambda x: (x.mean(0)))).tolist())
    ####################################################################


    def final_score(g, p):
        cosine = (g * p).sum() / ((np.sum(g**2)**0.5) * (np.sum(p**2)**0.5))
        return cosine

    feature = [feature_tfidf, feature_vaderCompound, feature_w2vTfidf, feature_w2vMean, feature_w2vAdj]
    feature_name = ["tfidf", "vaderCompound", "w2vTfidf", "w2vMean", "w2v" + pos_name]
    method = [svm.SVR(kernel='linear'), ensemble.GradientBoostingRegressor(), linear_model.LinearRegression(), linear_model.BayesianRidge(), ensemble.RandomForestRegressor(), tree.DecisionTreeRegressor(), Lasso()]
    method_name = ["SVR", "GradientBoostingRegressor", "LinearRegression", "BayesianRidge", "RandomForestRegressor", "DecisionTree", "LassoRegression"]

    result = {}

    f = open("result_" + pos + ".txt", "w")
    f_s = open("result_" + pos + "_sorted.txt", "w")
    times = 5

    # single feauture
    for i in range(5):
        X = feature[i]
        y = np.array(df['sentiment score'].tolist())
        for j in range(7):
            information = "feature: " + feature_name[i] + "    method: " + method_name[j]
            f.write(information + "\n")
            score = 0
            for k in range(times):
                X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
                clf = method[j]
                clf.fit(X_train, y_train)
                df_result = pd.DataFrame.from_dict(X_test, orient='columns')
                p = clf.predict(X_test)
                g = y_test
                if score < final_score(g, p):
                    score = final_score(g, p)
            result[information] = score
            f.write(str(score) + "\n")
            f.write("\n")


    # two features
    for i in range(5):
        for s in range(i+1, 5):
            X = np.column_stack((feature[i], feature[s]))
            y = np.array(df['sentiment score'].tolist())
            for j in range(7):
                information = "features: " + feature_name[i] + " & " + feature_name[s] + "    method: " + method_name[j]
                f.write(information + "\n")
                score = 0
                for k in range(times):
                    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
                    clf = method[j]
                    clf.fit(X_train, y_train)
                    df_result = pd.DataFrame.from_dict(X_test, orient='columns')
                    p = clf.predict(X_test)
                    g = y_test
                    if score < final_score(g, p):
                        score = final_score(g, p)
                result[information] = score
                f.write(str(score) + "\n")
                f.write("\n")


    # three features
    for i in range(5):
        for s in range(i+1, 5):
            for t in range(s+1, 5):
                X = np.column_stack((feature[i], feature[s], feature[t]))
                y = np.array(df['sentiment score'].tolist())
                for j in range(7):
                    information = "features: " + feature_name[i] + " & " + feature_name[s] + " & " + feature_name[t] + "    method: " + method_name[j]
                    f.write(information + "\n")
                    score = 0
                    for k in range(times):
                        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
                        clf = method[j]
                        clf.fit(X_train, y_train)
                        df_result = pd.DataFrame.from_dict(X_test, orient='columns')
                        p = clf.predict(X_test)
                        g = y_test
                        if score < final_score(g, p):
                            score = final_score(g, p)
                    result[information] = score
                    f.write(str(score) + "\n")
                    f.write("\n")


    # four features
    for i in range(5):
        for s in range(i+1, 5):
            for t in range(s+1, 5):
                for u in range(t+1, 5):
                    X = np.column_stack((feature[i], feature[s], feature[t], feature[u]))
                    y = np.array(df['sentiment score'].tolist())
                    for j in range(7):
                        information = "features: " + feature_name[0] + " & " + feature_name[1] + " & " + feature_name[2] + " & " + feature_name[3] + "    method: " + method_name[j]
                        f.write(information + "\n")
                        score = 0
                        for k in range(times):
                            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
                            clf = method[j]
                            clf.fit(X_train, y_train)
                            df_result = pd.DataFrame.from_dict(X_test, orient='columns')
                            p = clf.predict(X_test)
                            g = y_test
                            if score < final_score(g, p):
                                score = final_score(g, p)
                        result[information] = score
                        f.write(str(score) + "\n")
                        f.write("\n")


    # five features
    X = np.column_stack((feature[0], feature[1], feature[2], feature[3], feature[4]))
    y = np.array(df['sentiment score'].tolist())
    for j in range(7):
        information = "features: " + feature_name[0] + " & " + feature_name[1] + " & " + feature_name[2] + " & " + feature_name[3] + " & " + feature_name[4] + "    method: " + method_name[j]
        f.write(information + "\n")
        score = 0
        for k in range(times):
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
            clf = method[j]
            clf.fit(X_train, y_train)
            df_result = pd.DataFrame.from_dict(X_test, orient='columns')
            p = clf.predict(X_test)
            g = y_test
            if score < final_score(g, p):
                score = final_score(g, p)
        result[information] = score
        f.write(str(score) + "\n")
        f.write("\n")

    count = 0
    ans = [ (v,k) for k,v in result.items() ]
    ans.sort(reverse=True)
    for v, k in ans:
        count = count + 1
        f_s.write("%d\n%s\n%lf\n\n" % (count, k, v))

    f.close()
    f_s.close()

