import nltk
import random
from nltk.corpus import movie_reviews
import pickle

#nltk.download()

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
#随机排序一下
random.shuffle(documents)
print(documents[1])
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#统计每个词的出现频率
all_words = nltk.FreqDist(all_words)

#取出最常出现的15个词
print(all_words.most_common(15))

#查看stupid词出现的次数
print(all_words['stupid'])

word_features = list(all_words.keys())[:3000]
print(word_features)

#word_features，它包含了前 3000 个最常用的单词。 接下来，我们将建立一个简单的函数，在我们的正面和负面的文档中找到这些前 3000 个单词，将他们的存在标记为是或否：
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

#使用朴素贝叶斯分类算法训练
classifier = nltk.NaiveBayesClassifier.train(training_set)

#使用测试集验证准确度
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

#显示特征最明细的15个词，每一个词的负面到正面的出现几率
classifier.show_most_informative_features(15)

#使用pickle，保存分类器，在脚本当前目录下
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)#save_classifier指定保存分类器目录
save_classifier.close()

#加载已经保存的分类器
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)

#使用保存的分类器，测试集验证准确度
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier_f.close()

#引入sklearn，进化的贝叶斯算法
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
MnbClassifier = SklearnClassifier(MultinomialNB())
MnbClassifier.train(training_set)
print("MnbClassifier accuracy percent:",(nltk.classify.accuracy(MnbClassifier, testing_set))*100)


BnbClassifier = SklearnClassifier(BernoulliNB())
BnbClassifier.train(training_set)
print("BnbClassifier accuracy percent:",(nltk.classify.accuracy(BnbClassifier, testing_set))*100)

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

LogisticRegressionClassifier = SklearnClassifier(LogisticRegression())
LogisticRegressionClassifier.train(training_set)
print("LogisticRegressionClassifier accuracy percent:",(nltk.classify.accuracy(LogisticRegressionClassifier, testing_set))*100)

SGDCClassifier = SklearnClassifier(SGDClassifier())
SGDCClassifier.train(training_set)
print("SGDCClassifier accuracy percent:",(nltk.classify.accuracy(SGDCClassifier, testing_set))*100)

SVCClassifier = SklearnClassifier(SVC())
SVCClassifier.train(training_set)
print("SVCClassifier accuracy percent:",(nltk.classify.accuracy(SVCClassifier, testing_set))*100)

LSVCClassifier = SklearnClassifier(LinearSVC())
LSVCClassifier.train(training_set)
print("LSVCClassifier accuracy percent:",(nltk.classify.accuracy(LSVCClassifier, testing_set))*100)

NSVCClassifier = SklearnClassifier(NuSVC())
NSVCClassifier.train(training_set)
print("NSVCClassifier accuracy percent:",(nltk.classify.accuracy(NSVCClassifier, testing_set))*100)
