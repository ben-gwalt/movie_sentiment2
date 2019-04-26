import re
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

os.chdir('C:/Anaconda3/projects/movie_sentiment/movie_sentiment2')

reviews_train = []
reviews_test = []

for line in open('movie_data/full_train.txt', 'r', encoding="utf8"):
    reviews_train.append(line.strip())

for line in open('movie_data/full_test.txt', 'r', encoding="utf8"):
    reviews_test.append(line.strip())

REP_NOSPACE = re.compile('[.;:!\'?,\"()\[\]]')
REP_SPACE = re.compile('(<br\s*/><br\s*/>)|(\-)|(\/)')


def preprocess_rev(reviews):
    reviews = [REP_NOSPACE.sub('', line.lower()) for line in reviews]
    reviews = [REP_SPACE.sub(' ', line.lower()) for line in reviews]

    return reviews


reviews_train_clean = preprocess_rev(reviews_train)
reviews_test_clean = preprocess_rev(reviews_test)

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)


target = [1 if i <12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:

    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, lr.predict(X_val))))

final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
print('Final Accuracy: {}'.format(accuracy_score(target, final_model.predict(X_test))))

feature_to_coef = {word: coef for word, coef in zip(
    cv.get_feature_names(), final_model.coef_[0])
}

for best_positive in sorted(feature_to_coef.items(),
                            key=lambda x: x[1],
                            reverse=True)[:5]:
    print(best_positive)

for best_negative in sorted(feature_to_coef.items(),
                            key=lambda x: x[1])[:5]:
    print(best_negative)
