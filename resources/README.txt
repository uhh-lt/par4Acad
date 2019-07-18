Dataset for Ipar4Acad:Informal Word Identification and Paraphrasing for Academic Text
=====================================================================================

The zip file contains the following folders:
- academic
- aclacngrams
- iwi
- nonacademic
- paraphrase


academic folder
---------------
Contains the keyphrases extracted through the various techniques. The .csv files have the column headers
academic_freq : Frequency in the ACL corpus
non_academic_freq : Frequency in the Amazon Full Review dataset
phrase : Extracted phrase
ratio : The ratio of the academic frequency to the non-academic frequency

aclacngrams
-----------
These folder contains the most frequent ngrams (2-4) from the ACL anthology reference corpus. The file is formated as:
 Ngram TAB [POS TAG] TAB Frequency
 
 Example
 based on        [VBN, IN]       75158
 

iwi
---
The train and test files contain the following columns
freq_beautiful : Frequency of the lemma in the Beautiful Data Corpus
freq_coca_general : Frequency of the lemma in COCA
freq_acl : Frequency of the lemma in the ACL Anthology corpus
cos_target : The cosine similarity between the lemma and the target sentence
euclidean_distance : The Euclidean distance between the lemma and the target sentence
posMASC : POS tag of the word (from the CoInCo dataset)
posMASC_le : Label Encoded form of the column 'posMASC'
is_problematic : 1 if less than two annotators entered a substitute (otherwise : 0)
word_length : Word Length
count_vowel : The number of vowels in the lemma
y : Target variable (1 : Informal, 0 : Formal)

paraphrase
----------
The input format of the training and test files are the same as for SVM-rank (see https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html for further details).

nonacademic folder
---------------
Contains the keyphrases extracted through the various techniques. The .csv files have the column headers
academic_freq : Frequency in the ACL corpus
non_academic_freq : Frequency in the Amazon Full Review dataset
phrase : Extracted phrase
ratio : The ratio of the non-academic frequency to the academic frequency
