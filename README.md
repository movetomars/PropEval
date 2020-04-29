# UNTLing at SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles

This is the system described in the titular paper, to appear at COLING 2020, by [Maia Petee](https://github.com/movetomars) and [Alexis Palmer](https://github.com/alexispalmer).

### Installing

Make sure you have Scikit-Learn and spaCy installed in your development environment.

```
pip install sklearn
pip install spacy
```

You'll also want to take advantage of the pretrained GloVe vectors that we use by making sure you have spaCy's en_core_web_lg model installed.

```
python -m spacy download en_core_web_lg
```

## Running the classifiers

To run the SI (Span Identification) classifier, do the following (warning: you might need some heavy-duty computational power if you don't want to run into a memory error):

```
python mp-SI.py
```
You will receive the evaluation results printed to standard output.

To run the TC (Technique Classification) classifier, run the following command:

```
python mp-TC.py
```
The output of this classifier is a tab-delimited file that includes all propagandistic spans along with their predicted labels. This output template was made for submission to the official SemEval Task 11 site. 


## Deployment

Once downloaded, be sure to unzip the "datasets" and "tools" tarballs and store the directories in the same folder as the two programs.


## Acknowledgments

* Many thanks to the [UNT High-Performance Computing Center](https://github.com/gmihaila/unt_hpc) for the computing resources that made project development possible.
