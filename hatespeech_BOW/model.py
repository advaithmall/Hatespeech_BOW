from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
import sklearn
import numpy as np
import pandas as pd


train = pd.read_csv("train.csv")
test = pd.read_csv("file.csv")

test.tail()
#print(type(e))
sum(train["label"] == 0)
sum(train["label"]==1)


#set up punctuations we want to be replaced
no_space = re.compile(
    "(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
need_space = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

import preprocessor as pre
def cleaning_data(data_frame):
    fin_out = []
    for line in data_frame:
        temp = pre.clean(line)
        temp = no_space.sub("",temp.lower())
        temp = no_space.sub(" ",temp)
        fin_out.append(temp)
    return fin_out

print("Cleaning!")
train_tweet = cleaning_data(train["tweet"])
train_tweet = pd.DataFrame(train_tweet)

train["clean_tweet"] = train_tweet
#train.head(10)

test_tweet = cleaning_data(test["tweet"])
test_tweet = pd.DataFrame(test_tweet)
test["clean_tweet"] = test_tweet
#test.tail(10)
#print(len(test_tweet))

y = train.label.values
x_train, x_test, y_train, y_test = train_test_split(train.clean_tweet.values, y,
                                                    stratify=y,
                                                    random_state=1,
                                                    test_size=0.01, train_size=0.99, shuffle=True)

x_testf = np.array(test_tweet)
x_test=x_testf.flatten()
print("Training! (may take a minute)")
vectorizer = CountVectorizer(binary=True, stop_words='english')
vectorizer.fit(list(x_train) + list(x_test))
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)

svm = svm.SVC(kernel='linear', probability=True)
prob = svm.fit(x_train_vec, y_train).predict_proba(x_test_vec)
y_pred_svm = svm.predict(x_test_vec)

output = open("output.txt", "w")
for i in range(0, len(y_pred_svm)):
    if (y_pred_svm[i] == 1):
        print("HATE FOUND", " ---> ", x_test[i], file=output)
print("DONE!, check output.txt")
