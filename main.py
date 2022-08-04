import csv
import re
import nltk
import numpy as np
from numpy import reshape
from scipy.io import savemat, loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self, num_input, num_layer, num_class):
        self.num_class = num_class
        self.num_input = num_input
        self.model = self.create_dense([32] * num_layer)

    def create_dense(self, layer_sizes):
        model = Sequential()
        model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(self.num_input, )))
        for s in layer_sizes[1:]:
            model.add(Dense(s, activation='sigmoid'))
        model.add(Dense(self.num_class, activation='softmax'))
        return model

    def fit(self, X_train, y_train, batch_size=128, num_epochs=5):
        self.model.summary()
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=.1)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

        #plot_confusion_matrix(self.model, X_test, y_test)
    def score (self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=False)
        print(f'Test accuracy of Neural network: {accuracy:.3}')


def prepareDataset():
    textList = []
    emotionList = []
    with open('../src/datasets/tweet_emotions.csv', 'r') as inputFile:
        data = csv.reader(inputFile)
        count = 0
        for row in data:
            count += 1
            if count == 1:
                continue
            emotion = row[1]
            text = row[2]
            text = ' '.join(formatText(text))
            if emotion not in emotionList:
                emotionList.append(emotion)
            textList.append([emotion, text])

        inputFile.close()
    return emotionList, textList

def processData(emotionList, textList):
    X = []
    y = []
    dictionary = {}
    with open('vocab.txt') as inputFile:
        for row in inputFile.readlines():
            row = row.split(' ')
            dictionary[row[1]] = {'index': int(row[0]), 'count': int(row[2])}
        inputFile.close()
    count = 0
    for text in textList:
        emotion = text[0]
        content = text[1]
        content = content.split(' ')
        y.append(emotionList.index(emotion))
        X.append([0]*5000)
        for word in content:
            if word in dictionary:
                X[count][dictionary[word]['index']] = 1
        count += 1

    return X, y

def createVocab():
    textList = []
    with open('../src/datasets/tweet_emotions.csv', 'r') as inputFile:
        data = csv.reader(inputFile)
        count = 0
        for row in data:
            count += 1
            if count == 1:
                continue
            emotion = row[1]
            text = row[2]
            textList.append([emotion, text])
        inputFile.close()
    dictionary = {}
    vocabList = []
    for line in textList:
        text = line[1]
        formatted_text_list = formatText(text)
        for word in formatted_text_list:
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] += 1
    for key in dictionary:
        vocabList.append((key, dictionary[key]))
    vocabList.sort(key=lambda x: x[1], reverse=True)
    vocabList = vocabList[0:5000]
    vocabList.sort(key=lambda x: x[0])
    count = 0
    with open('vocab.txt', 'w') as outputFile:
        for element in vocabList:
            outputFile.write(str(count))
            outputFile.write(' ' + element[0] + ' ' + str(element[1]) + '\n')
            count += 1
        outputFile.close()

def formatText(text):
    text = text.lower()
    #remove html tag
    clean = re.compile('<[^<>]+>')
    text = re.sub(clean, ' ', text)
    #remove number
    clean = re.compile('[0-9]+')
    text = re.sub(clean, 'number', text)
    #remove url
    clean = re.compile('(http|https)://[^\s]*')
    text = re.sub(clean, 'httpaddr', text)
    #remove email address
    clean = re.compile('[^\s]+@[^\s]+')
    text = re.sub(clean, 'emailaddr', text)
    #remove money
    clean = re.compile('[$]+')
    text = re.sub(clean, 'dollar', text)
    #remove @ tag
    clean = re.compile('@[^\s]+')
    text = re.sub(clean, '', text)
    #split text using multiple delimiters
    text = re.split(r'[@$/#.\-:+&*=?!\[\]\\\(){},\'\'">_<;%\s]\s*', text)
    wordList = []
    for word in text:
        clean = re.compile('^a-zA-Z0-9')
        word = re.sub(clean, '', word)
        word = word.strip()

        if(len(word) > 2):
            #word = nltk.SnowballStemmer(language='english').stem(word)
            word = nltk.LancasterStemmer().stem(word)
            #word = nltk.PorterStemmer().stem(word)
        if (len(word) < 1):
            continue
        if(len(word) == 1 and word != 'i' and word != 'a' and word != 'u'):
            continue
        wordList.append(word)
    return wordList

def initialize():
    emotionList, textList = prepareDataset()
    createVocab()

    X, y = processData(emotionList, textList)

    X = np.array(X)
    y = np.array(y)

    data = {
        'X': X,
        'y': y
    }
    savemat('dataset.mat', data)

def runLogisticRegression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()
    print(f'Test accuracy of Logistic Regression: {score:.3f}')

def runNeuralNetwork(X_train, y_train, X_test, y_test, num_class):
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)
    model = NeuralNetwork(X_train.shape[1], 1, num_class)
    model.fit(X_train, y_train, num_epochs=1000)
    model.score(X_test, y_test)

def runSVM(X_train, y_train, X_test, y_test):
    model = svm.SVC(max_iter=1000)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()
    print(f'Test accuracy of SVM: {score:.3f}')

#initialize()

#print(formatText('academical'))
#print(formatText('academy'))


data = loadmat('dataset.mat')

X, y = data['X'], data['y']
y = reshape(y, (y.shape[1], ))
num_class = np.unique(y).shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('Training...')

#runLogisticRegression(X_train, y_train, X_test, y_test)
runNeuralNetwork(X_train, y_train, X_test, y_test, num_class)
#runSVM(X_train, y_train, X_test, y_test)




