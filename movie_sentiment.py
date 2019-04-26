import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

os.chdir('C:\\Anaconda3\\projects\\movie_sentiment')

data = pd.read_csv('train.tsv', sep='\t', na_filter=False)
data.head()

# LABELS
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive
data['Sentiment'].value_counts()

sentiment_count = data.groupby('Sentiment').count()
plt.bar(sentiment_count.index.values, sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Reviews')
plt.show()

# removing unwanted characters
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
# cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1), tokenizer=token.tokenize)
# text_counts = cv.fit_transform(data['Phrase'])
en_stopwords = set(stopwords.words("english"))


tf = TfidfVectorizer(lowercase=True,
                     tokenizer=token.tokenize,
                     analyzer='word',
                     stop_words=en_stopwords,
                     ngram_range=(1, 1))


# text_tf = tf.fit_transform(data['Phrase'])

X_train, X_test, y_train, y_test = train_test_split(data['Phrase'], data['Sentiment'], test_size=0.3, random_state=1)

kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

np.random.seed(1)

n_estimators=10
svc = BaggingClassifier(SVC(probability=True, kernel="linear", class_weight="balanced"),
                        max_samples=1.0 / n_estimators, n_estimators=n_estimators)


pipeline_svc = Pipeline([('tf', tf), ('svc', svc)])


grid_svc = GridSearchCV(pipeline_svc,
                        param_grid={'base_estimator__C': [1,10]},
                        cv=kfolds,
                        scoring='roc_auc',
                        verbose=50,
                        n_jobs=-1)

param = param_grid = {
 'bootstrap': [True, False],
 'bootstrap_features': [True, False],
 'n_estimators': [5, 10, 15],
 'max_samples' : [0.6, 0.8, 1.0],
 'base_estimator__bootstrap': [True, False],
 'base_estimator__n_estimators': [100, 200, 300],
 'base_estimator__max_features' : [0.6, 0.8, 1.0]
}



grid_svc = GridSearchCV(pipeline_svc,
                        param_grid=param,
                        cv=kfolds,
                        scoring='roc_auc',
                        verbose=50,
                        n_jobs=-1)

# sorted(grid_svc.get_params().keys())

grid_svc.fit(X_train, y_train)
grid_svc.score(X_test, y_test)

grid_svc.best_params_
grid_svc.best_score_
