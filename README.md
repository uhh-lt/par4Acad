# par4Acad

We present the first system that automatically edits a text so that it better adheres to the academic style of writing. The pipeline consists of an informal word identification (IWI), academic  candidate  paraphrase generation and paraphrase ranking  components. In addition to existing academic resources, such as the COCA academic word list, the New Academic Word List and the Academic Collocation List, we also explore how to build such resources that would be used to automatically identify informal or  non-academic words or phrases. The resources are compiled using different generic approaches that can be extended for different domains and languages. To generate candidates and rank them in context, we have used the PPDB and WordNet paraphrase resources.  We work with the Concepts in  Context (CoInCO) ”All-Words” lexical substitution dataset both for the informal word identification and paraphrase generation experiments. Our informal word identification component is able to perform 82.04% in F1, a significant improvement over the performance on the baseline systems. In addition to building the first informal word identification and paraphrasing system for academic writing, we also present the first generic system that can be used to build academic resources.

## Compile academic resources

1. Compute the n-grams from the academic corpus (ACL anthology)
```bash
$ python academic_ngrams.py
usage: academic_ngrams.py [-h] [--raw_academic_corpus RAW_ACADEMIC_CORPUS]
                          --text_academic_corpus TEXT_ACADEMIC_CORPUS
                          --serialize_output SERIALIZE_OUTPUT
```

2. Compute the n-grams from the non-academic corpus (Amazon Review Full Score Dataset)
```bash
$ python non_academic_ngrams.py
usage: non_academic_ngrams.py [-h] --text_non_academic_corpus TEXT_NON_ACADEMIC_CORPUS 
                              --serialize_output SERIALIZE_OUTPUT
```

3. Compile the academic/non-academic keyword list. The keyword list could be build based on TF-IDF (or could be based on [EmbedRank](https://github.com/swisscom/ai-research-keyphrase))
```bash
$ python keyword_model.py
usage: keyword_model.py [-h] [--raw_academic_corpus RAW_ACADEMIC_CORPUS]
                        --text_academic_corpus TEXT_ACADEMIC_CORPUS
                        --serialize_output SERIALIZE_OUTPUT
```
This would compile the keyword list based on the COCA criteria - retain those phrases that occur at least 50% more frequently in the academic portion of the corpora than would otherwise be expected. In other words, the ratio of the academic frequency of a term to it is non-academic frequency should be 1.50 or higher.

## Dataset for Informal Word Identification (IWI) and Paraphrasing Components

Wed derive our dataset from a lexical substitution dataset called Concepts In Context ([CoInCo](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/coinco.html)). The CoInCo dataset is an All-Words lexical substitution dataset, where all the words that could be substituted are manually annotated. A total of 1,608 train and 866 test sentences are compiled out of 2,474 sentenecs from the CoInCo dataset. 

We automatically generated an IWI dataset as follows. For each non-academic target word, we determine if the substitution candidate includes atleast one academic word. If so, it is labelled as **informal**, otherwise it is labelled as **formal**. All academic target words and all words without substitution candidates are labelled **formal**.

To generate **non-academic** to **academic** word pairs for paraphrasing we have included only those word pairs that where 1) the target word is non-academic, 2) the substitution candidate is academic, 3) the target word has higher word frequency than the substitute candidate in our academic resources. The dataset is prepared with 4 candidates for each informal target, where 2 candidates are academic and 2 candidates are non-academic. When we do not have appropriate candidates we extract further candidates from WordNet and PPDB.

## Informal Word Identification (IWI) Model

We have trained a few [classfiers](IWI.ipynb) provided by scikit-learn with the following features:

1. **Word Frequency** : We use the word frequencies 1) in [Beatiful Data](https://norvig.com/ngrams/) 2) in [COCA](https://www.english-corpora.org/coca/) general list 3) in [ACL anthology](https://acl-arc.comp.nus.edu.sg/) corpus.

2. **Word Embedding** : We have used [GloVe](https://nlp.stanford.edu/projects/glove/) to compute the cosine similarity between the word and the sentence. We also explore the option of using Euclidean distance between the word and the sentence while training the classifier.

3. **Part of Speech (POS) Tag and Word Level features** : We have used the word length and the number of vowels as features while training the classifier.

## Paraphrase Ranking Model

In order to rank the best candidates for academic rewriting, we have followed the learning to rank approach where the candidates are ranked baed on relevance score. The number of annotators selecting the given candidate is considered as the relevance score. The deep learning model provided by [tensorflow/ranking](https://github.com/tensorflow/ranking) is used o build the paraphrase ranking model.
