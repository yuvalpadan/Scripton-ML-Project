from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
import numpy as np
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

X = pickle.load(open("X3", "rb"))
y_train = pickle.load(open("y_train3", "rb"))
knn_model = pickle.load(open('knn_model3', 'rb'))
regression = pickle.load(open('regression3', 'rb'))
vectorizer = pickle.load(open("tfidf_vectorizer3", "rb"))

# N = 30
# for Product, category_id in sorted([("U", 0), ("18+", 4), ]):
#     features_chi2 = chi2(X, y_train == category_id)
#     indices = np.argsort(features_chi2[0])
#     feature_names = np.array(vectorizer.get_feature_names())[indices]
#     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#     print("# '{}':".format(Product))
#     print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
#     print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
#
model = LinearSVC()
model.fit(X, y_train)
N = 200
for Product, category_id in sorted([ ("18", 4)]):
    indices = np.argsort(model.coef_[category_id])
    feature_names = np.array(vectorizer.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
    print("# '{}':".format(Product))
    print("  . Top unigrams:\n       . {}".format('\n'.join(unigrams)))
    print("  . Top bigrams:\n       . {}".format('\n'.join(bigrams)))



# documents = [
#     "I'm a killer. A murdering bastard, you know that. And there are consequences to breaking the heart of a murdering bastard."
# ]
#
# document_names = ['Doc {:d}'.format(i) for i in range(len(documents))]
#
# def get_tfidf(docs, ngram_range=(1,1), index=None):
#     vect = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
#     tfidf = vect.fit_transform(documents).todense()
#     return pd.DataFrame(tfidf, columns=vect.get_feature_names(), index=index).T
#
# print(get_tfidf(documents, ngram_range=(1,2), index=document_names))
