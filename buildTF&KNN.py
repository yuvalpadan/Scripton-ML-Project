import sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import html2text
import requests
import string
import pickle
import xlrd
import re


data_file = "‏‏moviesData3.xlsx"


# Cleaning the text sentences
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in set(stopwords.words('english'))])
    punc_free = ''.join(ch for ch in stop_free if ch not in set(string.punctuation))
    normalized = " ".join(WordNetLemmatizer().lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y


# open the file
xlFile = xlrd.open_workbook(data_file)
sheet = xlFile.sheet_by_index(0)
sheet_len = len(sheet.col_values(0))
y_train = [100.] * sheet_len
train_clean_scripts = []

# reads the scripts
for line in range(1, sheet_len):
    rate = sheet.cell_value(line, 2)
    scriptUrl = sheet.cell_value(line, 3)
    try:
        request = requests.get(scriptUrl, timeout=10)
        script = html2text.HTML2Text().handle(request.text).strip()
        cleaned_script = clean(script)
        cleaned_script = ' '.join(cleaned_script)
        train_clean_scripts.append(cleaned_script)
        y_train[line - 1] = sheet.cell_value(line, 2)
    except:
        print(" # BAD SCRIPT: " + sheet.cell_value(line, 0), str(line + 1))
        continue
    print("The movie \"" + sheet.cell_value(line, 0) + "\" was added (line " + str(line + 1) + ")")

y_train = list(filter(100..__ne__, y_train))
y_train = np.array(y_train)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, max_df=50000, norm='l2', encoding='latin-1',
                        ngram_range=(1, 2), stop_words='english')
X = tfidf.fit_transform(train_clean_scripts)
pickle.dump(tfidf, open("tfidf_vectorizer3", "wb"))
pickle.dump(X, open("X3", "wb"))
pickle.dump(y_train, open("y_train3", "wb"))

# KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
knn_model.fit(X, y_train)
pickle.dump(knn_model, open('knn_model3', 'wb'))

# Linear regression
regression = sklearn.linear_model.LinearRegression()
regression.fit(X, y_train)
pickle.dump(regression, open('regression3', 'wb'))
