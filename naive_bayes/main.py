import pandas as pd
import re
from collections import Counter
sample = pd.read_table("./resource/SMSSpamCollection",sep="\t",names=['label','content'])
print(sample.head())

sample['label'] = sample.label.map({'ham':0,'spam':1})
#print(sample.head())
#print(sample['content'][0])
sentances = ['this is a dog!','that is a desk','Tomorrow need to work']
sans_punctuation_documents = []

for sentance in sentances:

    sans_punctuation_documents.append(re.sub(r'[^a-zA-Z0-9]', ' ', sentance.lower()))

print(sans_punctuation_documents)
words = []
for sentance in sans_punctuation_documents:
    words.append(sentance.split(" "))
print(words)

frequency_list = []
for word in words:
    for key,val in Counter(word).items():
        frequency_list.append({key:val})
print(frequency_list)

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
print(count_vector)
count_vector.fit(sentances)
print(count_vector.get_feature_names_out())
sentances_array = count_vector.transform(sentances).toarray()
print(sentances_array)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sample['content'],
                                                    sample['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(sample.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)
#print(training_data)
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)
#print(testing_data)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))