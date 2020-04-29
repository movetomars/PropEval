train_folder = "datasets/train-articles" # check that the path to the datasets folder is correct,
dev_folder = "datasets/dev-articles"     # if not adjust these variables accordingly
test_folder = "datasets/test-articles"
train_labels_file = "datasets/train-task2-TC.labels"
dev_labels_file = "datasets/dev-task-TC.labels"
dev_template_labels_file = "datasets/dev-task-TC-template.out"
test_template_labels_file = "datasets/test-task-TC-template.out"
propaganda_techniques_file = "tools/data/propaganda-techniques-names-semeval2020task11.txt" # propaganda_techniques_file is in the tools.tgz file (download it from the team page)
task_TC_output_file = "maias-output-TC.txt"
test_output_file = "gold-labels-mp.txt"


# ############################################################################  #
# Maia's TC Implementation: Using a Logistic Regression Classifier to Label Propaganda #
# ############################################################################  #

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import os.path
from sklearn.metrics import confusion_matrix
import spacy
nlp = spacy.load('en_core_web_lg') # Load spacy model to process the document


def read_articles_from_file_list(folder_name):
    """
    Read articles from files matching patterns <file_pattern> from  
    the directory <folder_name>.
    """
    file_list = os.listdir(folder_name)

    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])

    for filename in sorted(file_list):
        with open(folder_name + "/" + filename, "r", encoding="utf-8") as f:
            article_id = os.path.basename(filename).split(".")[0][7:]
            article_id_list.append(article_id)
            articles[article_id] = f.read()
    return articles


def read_predictions_from_file(filename):
    """
    Reader for the gold file and the template output file. 
    Return values are four arrays with article ids, labels 
    (or ? in the case of a template file), and the beginning
    and ending of fragments.
    """
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_labels.append(gold_label)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends, gold_labels


def valence(file):
    """
    Short function to read the VAD Lexicon values in
    and create them in dictionary form for retrieval
    during feature extraction.
    """
    vad = {}
    with open(file) as f:
        for line in f.readlines():
            word, valence, arousal, dominance = line.rstrip('\n').split('\t')
            vad[word] = {'valence':valence,'arousal':arousal,'dominance':dominance}
    return vad
    

def compute_features(articles, articles_id, span_starts, span_ends):
    """
    Feature extraction!
    """
    textfeatures = []

    file_c = 0

    vad = valence('tools/NRC-VAD-Lexicon.txt')

    for article in articles:
        ### Progress counter ###
        if (file_c % 10) == 0:
            print("processing article", file_c)
        file_c += 1
        ### End Progress counter ###
        article_id = article

        content = articles[article]
        content = content.replace('\n', ' ')

        article = nlp(content)

        for id, start, end in zip(articles_id, span_starts, span_ends):
            if article_id == id:
                fragment = article.text[int(start):int(end)]
                prop = nlp(fragment)

                valencecount = 0
                valence_fragment = {}
                arousal_fragment = {}
                dominance_fragment = {}

                # Extracting the sentiment features for each token
                for token in prop:
                    if token.text in vad:
                        valencecount += 1
                        valence_fragment[token.text] = vad[token.text]['valence']
                        arousal_fragment[token.text] = vad[token.text]['arousal']
                        dominance_fragment[token.text] = vad[token.text]['dominance']

                intensity = valencecount/len(prop)

                # The fragment's document vector
                vector = prop.vector
                vector = str(vector).replace('\n', '')

                # Named Entities
                ents = list(prop.ents)

                features = {
                    'text' : str(prop.text.split(' ')),
                    'length': str(len(prop)),
                    'ents' : str(ents),
                    'enttypes' : str([ent.label_ for ent in ents]),
                    'semantic' : vector,
                    'intensity' : str(intensity),
                    'valence' : str(valence_fragment),
                    'arousal' : str(arousal_fragment),
                    'dominance' : str(dominance_fragment)
                }

                textfeatures.append(features)

    return textfeatures

with open(propaganda_techniques_file, "r") as f:
    propaganda_techniques_names = [ line.rstrip() for line in f.readlines() ]

### MAIN ###
# loading articles' content from *.txt files in the train folder
articles = read_articles_from_file_list(train_folder)

# loading gold labels, articles ids and sentence ids from files *.task-TC.labels in the train labels folder 
ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = read_predictions_from_file(train_labels_file)
print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))

# Using preprocessing to encode string features
vec = DictVectorizer(sparse=False)

train = vec.fit_transform(compute_features(articles, ref_articles_id, ref_span_starts, ref_span_ends))

clf = LogisticRegression(penalty='l2',
                           class_weight='balanced',
                           solver="lbfgs",
                           verbose=1,
                           max_iter=200)


clf.fit(train, train_gold_labels)

## Reading data from the development set
dev_articles = read_articles_from_file_list(dev_folder)
test_articles = read_articles_from_file_list(test_folder)
dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_labels_file)
test_article_ids, test_span_starts, test_span_ends, test_labels = read_predictions_from_file(test_template_labels_file)

## Computing the predictions on the development set and test set
dev = vec.transform(compute_features(dev_articles, dev_article_ids, dev_span_starts, dev_span_ends))
test = vec.transform(compute_features(test_articles, test_article_ids, test_span_starts, test_span_ends))

## Change the argument given to the .predict() method to either "test" and "dev"
predictions = clf.predict(dev)
dev_labels = np.asarray(dev_labels)

labels = list(clf.classes_)

## Writing predictions to file
with open(task_TC_output_file, "w") as fout:
    for article_id, prediction, span_start, span_end in zip(dev_article_ids, predictions, dev_span_starts, dev_span_ends):
        fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
print("Predictions written to file " + task_TC_output_file)

## Creating and printing confusion matrices, both normalized and not
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(dev_labels, predictions)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()