import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
myStopword = stopwords.words('english')

def noisyTextToCleanText(noisyText):
    words = nltk.word_tokenize(noisyText)

    #remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped_words = [w.translate(table) for w in words]
    words_without_stopwords = []

    # removing stopwords
    for word in stripped_words:
        if word not in myStopword:
            words_without_stopwords.append(word)

    # convert word list to sentence
    sentence = ""
    for word in words_without_stopwords:
        sentence += " " + word
    return sentence


# 1)create train df with positive review
fileNameList = os.listdir(r"F:\python codes\interview_codes\aclImdb_v1\aclImdb\train\pos")
os.chdir(r"F:\python codes\interview_codes\aclImdb_v1\aclImdb\train\pos")

i = 0
listDict = []
for fileName in fileNameList:
    if i==1000:
        break
    file = open(fileName,encoding="utf8")
    dict = {}
    fileContent = file.read()
    # if 'dumbest' in fileContent:
    #     print("found: pos",fileName,fileContent[:100])
    dict['review'] = noisyTextToCleanText(fileContent)
    dict['result'] = 'pos'
    listDict.append(dict)
    file.close()
    i = i+1


df_train = pd.DataFrame(listDict)


# 2)create train df with negative review
fileNameList = os.listdir(r"F:\python codes\interview_codes\aclImdb_v1\aclImdb\train\neg")
os.chdir(r"F:\python codes\interview_codes\aclImdb_v1\aclImdb\train\neg")

i=0
listDict = []
for fileName in fileNameList:
    if i==1000:
        break
    file = open(fileName,encoding="utf8")
    dict = {}
    fileContent = file.read()
    # if 'dumbest' in fileContent:
    #     print("found: neg ",fileName,fileContent[:100])
    dict['review'] = noisyTextToCleanText(fileContent)
    dict['result'] = 'neg'
    listDict.append(dict)
    file.close()
    i = i+1



df_train = df_train.append(listDict)

# now generate test data
# 3)create test df with positive review
fileNameList = os.listdir(r"F:\python codes\interview_codes\aclImdb_v1\aclImdb\test\pos")
os.chdir(r"F:\python codes\interview_codes\aclImdb_v1\aclImdb\test\pos")

i = 0
listDict = []
for fileName in fileNameList:
    if i==1000:
        break
    file = open(fileName,encoding="utf8")
    dict = {}
    fileContent = file.read()
    # if 'dumbest' in fileContent:
    #     print("found: pos ",fileName,fileContent[:100])
    dict['review'] = noisyTextToCleanText(fileContent)
    dict['result'] = 'pos'
    listDict.append(dict)
    file.close()
    i = i+1


df_test = pd.DataFrame(listDict)


# 4)create test df with negative review
fileNameList = os.listdir(r"F:\python codes\interview_codes\aclImdb_v1\aclImdb\test\neg")
os.chdir(r"F:\python codes\interview_codes\aclImdb_v1\aclImdb\test\neg")

i = 0
listDict = []
for fileName in fileNameList:
    if i==1000:
        break
    file = open(fileName,encoding="utf8")
    dict = {}
    fileContent = file.read()
    # if 'dumbest' in fileContent:
    #     print("found: neg ",fileName,fileContent[:100])
    dict['review'] = noisyTextToCleanText(fileContent)
    dict['result'] = 'neg'
    listDict.append(dict)
    file.close()
    i = i+1


df_test = df_test.append(listDict)

# all data(train+test) for getting common vector
df = df_train.append(df_test,ignore_index=False)


y_train = df_train['result'].values


from sklearn.feature_extraction import text
# countvector only takes 1d array
# need to provide the whole train-test data to generate vector that can hold any text
# this fit determines the length of each vector
# length of vectors must be equal for train and test and any new data
# if the new data has the
vectorizer = text.CountVectorizer(binary=False).fit(df['review'].values)
X_train = vectorizer.transform(df_train['review'].values)

y_test = df_test['result'].values

X_test = vectorizer.transform(df_test['review'].values)


# training the model on training set
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
# takes 2d data
gnb.fit(X_train.toarray(), y_train)

# takes 2d data
y_pred = gnb.predict(X_test.toarray())

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)





