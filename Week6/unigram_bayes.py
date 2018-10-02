import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class bayes():
    
    
    def __init__(self, data):
        
        # split into training and testing data
        self.train_data, self.test_data = train_test_split(data,
                                            random_state=42, train_size=.8)
        # convert into n grams
        self.train_data = [[item[0], self.unigrams(item[1])] for item in self.train_data]
        self.test_data = [[item[0], self.unigrams(item[1])] for item in self.test_data]
        
        # count unique n grams in training data
        flattened = [gram for message in self.train_data for gram in message[1]]
        self.unique = len(set(flattened))
        
        # init dicts
        self.trainPositive = {}
        self.trainNegative = {}
        # counters
        self.posGramCount = 0
        self.negGramCount = 0
        self.spamCount = 0
        # priors
        self.pA = 0
        self.pNotA = 0
        
    def unigrams(self, text):
        text = text.split(' ')
        grams = []
        for i in range(len(text)):
            gram = ' '.join(text[i:i+1])
            grams.append(gram)
        return grams 
    
    def train(self):
        
        for item in self.train_data:
            label = item[0]
            grams = item[1]
            if label == 1:
                self.spamCount += 1   
            for gram in grams:
                if label == 1:
                    self.trainPositive[gram] = self.trainPositive.get(gram, 0) + 1
                    self.posGramCount += 1
                else:
                    self.trainNegative[gram] = self.trainNegative.get(gram, 0) + 1
                    self.negGramCount += 1
                    
        self.pA = self.spamCount/float(len(self.train_data))
        self.pNotA = 1.0 - self.pA
        
    def classify(self, text, alpha=1.0):
        
        self.alpha = alpha
        isSpam = self.pA * self.conditionalText(text, 1)
        notSpam = self.pNotA * self.conditionalText(text, 0)
        if (isSpam > notSpam):
            return 1
        else:
            return 0
        
    def conditionalText(self, grams, label):
        result = 1.0
        for unigram in grams:
            result *= self.conditionalUnigram(unigram, label)
        return result
    
    def conditionalUnigram(self, unigram, label):
        alpha = self.alpha
        if label == 1:
            return ((self.trainPositive.get(unigram,0)+alpha) /
                    float(self.posGramCount+alpha*self.unique))
        else:
            return ((self.trainNegative.get(unigram,0)+alpha) /
                    float(self.negGramCount+alpha*self.unique))
            
    def evaluate_test_data(self):
        results = []
        for test in self.test_data:
            label = test[0]
            text = test[1]
            ruling = self.classify(text)
            if ruling == label:
                results.append(1) 
            else:
                results.append(0) 
                
        print("Evaluated {} test cases. {:.2f}% Accuracy".format(len(results), 100.0*sum(results)/float(len(results))))
        return sum(results)/float(len(results))


df = pd.read_csv('./data/spam.csv')
# label spam as 1, not spam as 0
df['v1'] = df['v1'].replace(["ham","spam"],[0,1])
data = df.values

unigram_bayes = bayes(data)
unigram_bayes.train()

unigram_bayes.evaluate_test_data()