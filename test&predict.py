import html2text
import requests
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pickle
import string
import re


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in set(stopwords.words('english'))])
    punc_free = ''.join(ch for ch in stop_free if ch not in set(string.punctuation))
    normalized = " ".join(WordNetLemmatizer().lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y


vectorizer = pickle.load(open("tfidf_vectorizer2", "rb"))
knn_model = pickle.load(open('knn_model2', 'rb'))
regression = pickle.load(open('regression2', 'rb'))



scripts_link_dict = dict()
scripts_link_dict["Nightmare-on-Elm-Street"] = "https://www.imsdb.com/scripts/Nightmare-on-Elm-Street,-A.html"
# scripts_link_dict["THE TRUMAN SHOW"] = "https://www.imsdb.com/scripts/American-Madness.html"
# scripts_link_dict["10 THINGS I HATE ABOUT YOU"] = "https://www.imsdb.com/scripts/Grand-Hotel.html"
# scripts_link_dict["MISERY"] = "https://www.imsdb.com/scripts/Forrest-Gump.html"
# scripts_link_dict["MIDNIGHT EXPRESS"] = "https://www.imsdb.com/scripts/Black-Swan.html"




test_scripts = list()
movie_names = list()
# test_scripts.append("I'm a killer. A murdering bastard, you know that. And there are consequences to breaking the heart of a murdering bastard.")
# movie_names.append("test")
for name, link in scripts_link_dict.items():
    try:
        request = requests.get(link, timeout=6)
        script = html2text.HTML2Text().handle(request.text).strip()
        test_scripts.append(script)
        movie_names.append(name)
    except:
        continue
    print("the script of \"" + name + "\" was added")

print("\ncalculating results...\n")

test_clean_sentence = []
for test in test_scripts:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+", "", cleaned)
    test_clean_sentence.append(cleaned)

Test = vectorizer.transform(test_clean_sentence)

true_test_labels = ['U', 'PG', '12+', '15+', '18+']
predicted_labels_knn = knn_model.predict(Test)
predicted_regression = regression.predict(Test)
score = knn_model.kneighbors(Test)

print(" ### KNN ###")
for i in range(len(test_scripts)):
    print(movie_names[i] + ": " + true_test_labels[np.int(predicted_labels_knn[i])])

print(" ####  LINEAR REGRESSION RESULTS  ####\n")
for i in range(len(test_scripts)):
    name = movie_names[i]
    print("\t" + movie_names[i] + " "*(30 - len(name)) + true_test_labels[np.int(predicted_regression[i])])




