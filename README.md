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
