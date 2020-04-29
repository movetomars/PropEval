dev_folder = "datasets/dev-articles" # check that the path to the datasets folder is correct, if not adjust these variables accordingly
train_folder = "datasets/train-articles"
train_labels = "datasets/train-labels-task1-span-identification"
dev_labels = "datasets/dev-labels-task1-span-identification"
outfile = "mp-output-SI.txt"

# ############################################################################  #
# Maia's SI Implementation: Using a Sequence Labeler to Identify Propaganda Fragments #
# ############################################################################  #

# Misc Imports

import os.path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# NLP Imports

import spacy
nlp = spacy.load('en_core_web_lg') # Load spacy model to process the document

# CRF-Specific Imports

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# loading articles' content and gold standard labels from *.txt files in the train and dev folders
def extract_files(folder, labels):
    """ Function that extracts the content and article ids from any given folder"""

    articles_content, articles_id = ([], [])
    labels_list = os.listdir(labels)
    # Creating dictionary of gold label indices, to feed into the next step of
    # labeling training data

    gold_labels = {}

    for item in labels_list:
        with open(labels + "/" + item, "r", encoding="utf-8") as f:
            gold_labels[item[7:16]] = []
            for line in f:
                line = line.rstrip('\n')
                target = line.split('\t')
                gold_labels[item[7:16]].append(target[1:])


    file_list = os.listdir(folder)

    for filename in file_list:
        with open(folder+"/"+filename, "r", encoding="utf-8") as f:
            text = f.read().replace('\n',' ')

            articles_content.append(text)
            articles_id.append(os.path.basename(filename).split(".")[0][7:])

    return articles_content, articles_id, gold_labels

def get_vector(token):

    try:
         vector=token.vector
    except:
        # if the word is not in vocabulary,
        # returns zeros array
        vector=np.zeros(300,)

    return vector

def labelit(folder, labels):
    """Labeling it."""

    articles_content, articles_id, gold_labels = extract_files(folder, labels)

    textlabels = []     # Initializing a dictionary that will contain tuples with tokens and gold standard labels

    file_c = 0  # This is just so you don't go nuts watching my slow code run
    for article_content, article_id in zip(articles_content, articles_id):
        ### Progress counter ###
        if (file_c % 10) == 0:
            print("processing article", file_c)
        file_c += 1
        ### End Progress counter ###

        doc = nlp(article_content)
        articlelabels = []

        # For each set of gold labels, set labeling parameters and assign label "P" or "N"
        for sent in doc.sents:
            for i, token in enumerate(sent, 0):
                vector = get_vector(token)

                for prop_fragment in gold_labels[article_id]:
                    start_fragment, end_fragment = (int(prop_fragment[0]), int(prop_fragment[1]))

                    if token.idx < start_fragment or token.idx >= end_fragment:
                        label = 'N'
                    elif token.idx == start_fragment:
                        label = 'B-P'
                    else:
                        label = 'I-P'

                if len(token) > 0:
                    prev, next = None, None

                    if i > 0:
                        prev = doc[i - 1]
                    if i < len(doc) - 1:
                        next = doc[i + 1]

                    features = [
                        'bias=%s' % 1.0,
                        'word=' + token.text,
                        'pos=' + token.tag_,
                        'word.dependency=' + token.dep_
                    ]

                    chunk_prev, chunk_next = None, None

                    if i < 5:
                        bow_beginning = sent[0:i]
                    else:
                        bow_beginning = sent[i - 5:i]

                    if len(bow_beginning) > 0:
                        chunk_prev = bow_beginning.vector
                        chunk_prev = str(chunk_prev).replace('\n', '')

                    if i < (len(sent)-5):
                        bow_end = sent[i+1:i+6]
                    else:
                        bow_end = sent[i+1:]

                    if len(bow_end) > 0:
                        chunk_next = bow_end.vector
                        chunk_next = str(chunk_next).replace('\n', '')

                    if prev is not None:
                        features.extend([
                            'prevfirstword='+ prev.text,
                            'prevwordpos=' + prev.tag_,
                            'bow_beginning=%s' % bow_beginning.text.split(' '),
                            'prevword.dependency=' + prev.dep_
                        ])
                    else:
                        features.append('BOS')

                    if next is not None:
                        features.extend([
                            'nextfirstword=' + next.text,
                            'nextwordpos=' + next.tag_,
                            'bow_end=%s' % bow_end.text.split(' '),
                            'nextword.dependency=' + next.dep_
                        ])

                    if chunk_prev is not None:
                        features.extend([
                            'semantic_left=%s' % chunk_prev,
                        ])

                    if chunk_next is not None:
                        features.extend([
                            'semantic_right=%s' % chunk_next,
                        ])
                    else:
                        features.append('EOS')

                    for iv, value in enumerate(vector):
                        features.append('v{}=%s'.format(iv) % value)

                    articlelabels.append((features, label))

        textlabels.append(articlelabels)
    print("Building model...")
    return textlabels


def tokenfeatures(doc, i):
    """ Extracting features per token"""
    features = doc[i][0]
    return features

def extract_features(doc):
    """ Extracting features per document"""
    return [tokenfeatures(doc, i) for i in range(len(doc))]

def get_labels(doc):
    """Retrieving labels for each training token"""
    return [label for (features, label) in doc]

def sent2tokens(doc):
    """ Retrieving the original words """
    return [features['word'] for (features, label) in doc]

textlabels = labelit(train_folder, train_labels)
devlabels = labelit(dev_folder, dev_labels)

X_train = [extract_features(doc) for doc in textlabels]
X_test = [extract_features(doc) for doc in devlabels]

y_train = [get_labels(doc) for doc in textlabels]
y_test = [get_labels(doc) for doc in devlabels]


#%%time
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.01,
    max_iterations=200,
    all_possible_transitions=True,
    verbose=True
)

crf.fit(X_train, y_train)

labels = list(crf.classes_)
y_pred = crf.predict(X_test)

metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])


def print_state_features(state_features):

    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

print("\nTop positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])

mlb = MultiLabelBinarizer()
encode_train = mlb.fit_transform(y_train)
encode_test = mlb.transform(y_test)
encode_predictions = mlb.transform(y_pred)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """Creating a confusion matrix for token labels"""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(encode_test.argmax(axis=1), encode_predictions.argmax(axis=1))
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
plt.savefig('SI-matrix.png')

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.savefig('SI-matrix-normalized.png')
plt.show()

