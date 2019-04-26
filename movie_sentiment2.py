import re
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

en_stopwords = set(stopwords.words("english"))


def remove_stop(corpus):
    removed_stop = []
    for review in corpus:
        removed_stop.append(' '.join(
            [word for word in review.split()
             if word not in en_stopwords])
        )
    return removed_stop


no_stop = remove_stop(reviews_train_clean)


def lemmatize(corpus):
    lemmatizer = WordNetLemmatizer()
    return[' '.join(
        [lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]


lemmatized = lemmatize(reviews_train_clean)

ngram_vec = CountVectorizer(binary=True, ngram_range=(1,2))
ngram_vec.fit(reviews_train_clean)
X = ngram_vec.transform(reviews_train_clean)
X_test = ngram_vec.transform(reviews_test_clean)

# tfidf_vec = TfidfVectorizer()
# tfidf_vec.fit(reviews_train_clean)
# X = tfidf_vec.transform(reviews_train_clean)
# X_test = tfidf_vec.transform(reviews_test_clean)

target = [1 if i <12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:

    svm = LinearSVC(C=c)
    svm.fit(X_train, y_train)
    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, svm.predict(X_val))))

final_model = LinearSVC(C=0.01)
final_model.fit(X, target)
print('Final Accuracy: {}'.format(accuracy_score(target, final_model.predict(X_test))))

feature_to_coef = {word: coef for word, coef in zip(
    ngram_vec.get_feature_names(), final_model.coef_[0])
    # tfidf_vec.get_feature_names(), final_model.coef_[0])
}

print(' ')
print('Strongest Postitive: ')
for best_positive in sorted(feature_to_coef.items(),
                            key=lambda x: x[1],
                            reverse=True)[:5]:
    print(str(best_positive))

print(' ')
print('Strongest Negative: ')
for best_negative in sorted(feature_to_coef.items(),
                            key=lambda x: x[1])[:5]:
    print(str(best_negative))



