dev_folder = "datasets/dev-articles" # check that the path to the datasets folder is correct, if not adjust these variables accordingly
train_folder = "datasets/train-articles"
train_labels = "datasets/train-labels-task1-span-identification"
propaganda_techniques_file = "tools/data/propaganda-techniques-names-semeval2020task11.txt" # propaganda_techniques_file is in the tools.tgz file (download it from the team page)
task_SI_output_file = "mp-output-SI.txt"

# ############################################################################  #
# Maia's SI Baseline: Using a Sequence Labeler to Identify Propaganda Fragments #
# ############################################################################  #

# Plotting Imports

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Misc Imports

from itertools import chain
import os.path

# NLP Imports

import nltk
import sklearn
import spacy
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# CRF-Specific Imports

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# loading articles' content and gold standard labels from *.txt files in the dev folder

train_file_list = os.listdir(train_folder)
train_labels_list = os.listdir(train_labels)
dev_file_list = os.listdir(dev_folder)

train_articles_content, train_articles_id, gold_labels = ([], [], {})
test_articles_content, test_articles_id = ([], [])

# Creating list of filenames for both train and test folders (identifying numbers)

for filename in train_file_list:
    with open(train_folder+"/"+filename, "r", encoding="utf-8") as f:
        train_articles_content.append(f.read())
        train_articles_id.append(os.path.basename(filename).split(".")[0][7:])

for filename in dev_file_list:
    with open(dev_folder + "/" + filename, "r", encoding="utf-8") as f:
        test_articles_content.append(f.read())
        test_articles_id.append(os.path.basename(filename).split(".")[0][7:])

# Creating dictionary of gold label indices, to feed into training later

for labels in train_labels_list:
    with open(train_labels+"/"+labels, "r", encoding="utf-8") as f:
        gold_labels[labels[7:16]] = []
        for line in f:
            line = line.rstrip('\n')
            target = line.split('\t')
            gold_labels[labels[7:16]].append(target[1:])


def tokenfeatures(sent, i):
    token = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': token.text.lower(),
        'word[-3:]': token.text[-3:],
        'word[-2:]': token.text[-2:],
        'word.isupper()': token.text.isupper(),
        'word.istitle()': token.text.istitle(),
        'word.isdigit()': token.text.isdigit(),
        'word.ispunc()': token.pos_ == 'PUNCT',
        'postag': token.pos_,
    }

    sent_i = token.i - sent.start

    if sent_i > 0:
        word1 = sent[sent_i-1].text
        postag1 = sent[sent_i-1].pos_
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1
        })
    else:
        features['BOS'] = True

    if sent_i < len(sent)-1:
        word1 = sent[sent_i+1].text
        postag1 = sent[sent_i+1].pos_
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1
        })
    else:
        features['EOS'] = True

    return features


def documentfeatures(article_id):
    return [tokenfeatures(token, i) for i in range(len())]

def traininglabels():
    labels = {}

    with open(task_SI_output_file, "w") as fout:
        for article_content, article_id in zip(train_articles_content, train_articles_id):

            nlp = spacy.load('en') # Load spacy model to process the document
            doc = nlp(article_content)

            labels[article_id] = [] # Initializing list of gold standard labels for article in question

            # For each set of gold labels, set labeling parameters and assign label "P" or "N"
            for prop_fragment in gold_labels[article_id]:
                start_fragment, end_fragment = (int(prop_fragment[0]), int(prop_fragment[1]))
                # print('Propaganda fragment:',article_content[start_fragment:end_fragment])

                for token in doc:
                    if token.idx < start_fragment or token.idx >= end_fragment:
                        labels[article_id].append("N")
                    else:
                        labels[article_id].append("P")

    return labels

# def sent2labels(sent):
#     return [label for sentencefeatures(sent), label in sent]
#
# def sent2tokens(sent):
#     return [token for sentencefeatures(sent), label in sent]

# Defining training and testing data

# X_train = [sentencefeatures(art) for art in train_articles_content]
# y_train = [sent2labels(s) for art in train_articles_content]
#
# X_test = [sentencefeatures(s) for s in test_sents]
# y_test = [sent2labels(s) for s in test_sents]

# Creating our CRF classifier
#
# %%time
# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.1,
#     c2=0.1,
#     max_iterations=100,
#     all_possible_transitions=True
# )
# crf.fit(X_train, y_train)

# Baselines for sequence labeling (see CRF, Scikit Learn)
# Produce 0 or 1 (binary) labels per token using training data

# Steps :
#   1) Read content of training files into one large string (Done)
#   2) Compile huge dictionary of labeled bits from training labels folder (Done)
#   3) Initialize CRF with Scikit Learn
#   4) Train on propaganda fragments from dictionary - how?
#   5) Use spaCy to feed articles in dev folder to classifier in similar way
#   6) Profit???
