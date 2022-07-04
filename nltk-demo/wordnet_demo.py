from nltk.corpus import wordnet
#查找同义词
syns = wordnet.synsets("program")

print(syns[0].name())

#仅单词
print(syns[0].lemmas()[0].name())

#单词定义
print(syns[0].definition())

#单词示例
print(syns[0].examples())


#######################查找good的同义词、反义词##################
synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

#比较以下几组词的相似度
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

# 0.9090909090909091

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

# 0.6956521739130435

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))