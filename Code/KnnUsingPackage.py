# Importing all the required packages and libraries required for running the program
 
import pandas as pd 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from nltk.corpus import stopwords 
import re
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# Function to calculate the Euclidean Distance for the given points 
def eucli_distance(firstx,secondx):
    firstx = np.array(firstx)
    secondx = np.array(secondx)
    distance = 0
    for i in range(len(firstx)):
        distance = distance + (firstx[i]-secondx[i])**2
    return np.sqrt(distance)

# Function to predict the labels for the test data 
class custom_K_Nearest_Neighbors:
    def __init__(self, k, dist_metric=eucli_distance):
        self.k = k
        self.dist_metric = dist_metric
    
    def fit(self, xtrain, ytrain):
        self.xtrain = list(xtrain)
        self.ytrain = list(ytrain)

    def makePredictions(self, xtest):
        labelPredictions = []
        xtest=list(xtest)
        
        for i in range(len(xtest)):
            positiveCounter = 0
            negativeCounter = 0
            eDistances = []
            
            for j in range(len(self.xtrain)):
                eDis = eucli_distance(xtest[i],self.xtrain[j])
                eDistances.append((eDis, self.ytrain[j]))
            
            eDistances = sorted(eDistances, key=lambda x:x[0])
            slicedDistances = eDistances[:self.k]
        
            for l in range(len(slicedDistances)):
                if slicedDistances[l][1] == 1:
                    positiveCounter = positiveCounter+1 
                else:
                    negativeCounter = negativeCounter+1 
            
            if negativeCounter>positiveCounter:
                labelPredictions.append(-1)
            else:
                labelPredictions.append(1)  
        
        return labelPredictions


# Function to pre-process the data before passing it to the model
def textCleaning(rawData):

    letters_only = re.sub("[^a-zA-Z]", " ", rawData)

    wordsLower = letters_only.lower().split()     
    
    stopWords = set(stopwords.words("english"))                  
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = ''
    for word in wordsLower:
        if word not in stopWords:
            lemmatized_words += str(lemmatizer.lemmatize(word)) + ' '
    return lemmatized_words

# Reading the data using pandas function from csv file
dataset = pd.read_csv('../Data/train_file.csv', header = None)
test_dataset = pd.read_csv('../Data/test_file.csv', header = None)
dataset = dataset[:10000] # Slicing the data upto 100 values for easy processing
test_dataset = test_dataset[:10000]
X = dataset[1] # text data
y = dataset[0] # label data
T = test_dataset[0]

# print(">>> Before Cleaning:\n", X[0])
# print("\n")

# Cleaing the text data
cleanedData = []
t_cleaned_data =[]
for line in X:
     cleanedData.append(textCleaning(line))
for tline in T:
     t_cleaned_data.append(textCleaning(tline))
# print(">>> After Cleaning:\n", cleanedData[0])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(cleanedData, y, test_size=0.2, random_state=42)

# Applying Tfidf Vectorizer on the data after split
tfidf_transformer = TfidfVectorizer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_test_tfidf = tfidf_transformer.transform(X_test)
T_test_tfidf = tfidf_transformer.transform(t_cleaned_data).toarray()

# tsvd=TruncatedSVD(5)
# X_train_svd = tsvd.fit_transform(X_train_tfidf)
# X_test_svd = tsvd.ransform(X_test_tfidf)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_tfidf, y_train)
predictions = knn.predict(T_test_tfidf)

print("\nAccuracy own knn:{}".format(accuracy_score(predictions, y_test)))

# mstring =''
# for prediction in predictions:
#     mstring = mstring + '\n' + str(prediction)

# fi = open("TestDataPredictionsHW1.txt", "w") 
# print(mstring, file=fi)
# fi.close()