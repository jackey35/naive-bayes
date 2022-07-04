from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("omw-1.4")

#词性还原，例如：复数还原单数，比较级还原原形
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a")) # output good
print(lemmatizer.lemmatize("best", pos="a"))  #  output best??,wordnet没用匹配的词，就返回原词

print(lemmatizer.lemmatize("bigger", pos="a"))
print(lemmatizer.lemmatize("biggest", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))