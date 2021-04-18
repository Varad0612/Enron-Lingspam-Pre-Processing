import glob, csv
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from random import shuffle
import numpy

# Get contents of the entire corpus
def getCorpus(path):
    corpus = defaultdict(list)
    for x in ['ham', 'spam']:
        tmpPath = path + "/" + x + "/*"
        for name in glob.glob(tmpPath):
            f = open(name, "r")
            try:
                corpus[x].append(f.read())
            except:
                pass
            f.close()
    return corpus

# Get contents of the entire corpus
def getTestCorpus(path):
    corpus = defaultdict(list)
    for i in range(1,2):
        for x in ['ham', 'spam']:
            tmpPath = path + str(i) + "/" + x + "/*"
            for name in glob.glob(tmpPath):
                f = open(name, "r")
                try:
                    corpus[x].append(f.read())
                except:
                    pass
                f.close()
    return corpus

# Fit corpus to get the vocabulary
def fitCorpus(corpus):
    # Corpus is a dict with keys 'ham' and 'spam'
    # Create dictionary of vocab
    vectorizer = CountVectorizer(max_features = 4096, stop_words='english', min_df=50, binary=True)
    vectorizer.fit(corpus['ham'] + corpus['spam'])
    return vectorizer

# Tokenize the corpus and obtain the documents as feature vectors
def vectorize(vectorizer, corpus):
    # Corpus is a dict with keys 'ham' and 'spam'
    # Create dictionary of vocab
    hamDocs = vectorizer.transform(corpus['ham']).toarray()
    spamDocs = vectorizer.transform(corpus['spam']).toarray()
    print(hamDocs.shape)
    print(spamDocs.shape)
    print("Ham : " + str(len(hamDocs)) + "\n")
    print("Spam : " + str(len(spamDocs)) + "\n")

    # Append appropriate label and prepend bias term
    spam = []
    ham = []
    for i in range(len(hamDocs)):
        tmp = list(hamDocs[i])
        tmp.append(0)
        tmp.insert(0,1)
        ham.append(tmp)
    for i in range(len(spamDocs)):
        tmp = list(spamDocs[i])
        tmp.append(1)
        tmp.insert(0,1)
        spam.append(tmp)
    allDocs = ham + spam
    shuffle(allDocs)
    return allDocs

# Write the document vectors to csv files
def writeToCsv(outPath, corpus):
    f = open(outPath, "a", newline='')
    w = csv.writer(f, dialect='excel')
    for x in corpus:
        w.writerow(x)
    f.close()

    
trainCorpusPath = "./enron/TrainingSet"
trainsetPath = "./enron/train.csv"
testCorpusPath = "./enron/TestSet"
testSetPath = "./enron/test.csv"
traincorpus = getCorpus(trainCorpusPath)
print(len(traincorpus))
vectorizer = fitCorpus(traincorpus)
docVecs = vectorize(vectorizer, traincorpus)
writeToCsv(trainsetPath, docVecs)
docVecs = vectorize(vectorizer, getCorpus(testCorpusPath))
writeToCsv(testSetPath, docVecs)
