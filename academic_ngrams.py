import os, sys, re, tarfile, collections
import argparse
import pickle
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

class Academic:

    def __init__(self, acl_path, serialize_path):
        self.ACL_PATH = acl_path
        self.SERIALIZE = serialize_path
        self.unigrams_ctr = Counter()
        self.bigrams_ctr = Counter()
        self.trigrams_ctr = Counter()
        self.quadgrams_ctr = Counter()

    def clean_content(self, content, remove_stopwords=False, lemmatize_words=True):
        """Clean the dataset - remove stop words, lemmatize the word tokens
        :param content: The string that needs to be cleaned
        :type content: str
        :param remove_stopwords: default False
        :type remove_stopwords:  bool
        :param lemmatize_words: default True
        :type lemmatize_words:  bool
        """
        content = " ".join(re.findall(r"[a-zA-Z0-9]+", content)) # Remove special characters

        content = content.lower() # Lower case

        if(remove_stopwords):
            stop_words = set(stopwords.words('english')) # Remove stop words
        word_tokens = word_tokenize(content)

        if(lemmatize_words and not remove_stopwords):
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in word_tokens]

        if(lemmatize_words and remove_stopwords):
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in word_tokens if word not in stop_words]

        content = " ".join(text)
        return content

    def get_ngram(self, content, n):
        """Compute the n-grams using NLTK
        :param content: The string from which the n grams have to be computed
        :type content: str
        :param n: Specify whether 2, 3, 4 gram to be computed
        :type n: int
        """
        tokenized = content.split()
        es_ngrams = ngrams(tokenized, n)
        return list(es_ngrams)

    def update_unigram_counter(self, unigram):
        """Update the frequency counter
        :param unigram: List of unigrams
        :type unigram: list
        """
        for u in unigram:
            self.unigrams_ctr[u] += 1

    def update_bigram_counter(self, bigram):
        """Update the frequency counter
        :param bigram: List of bigrams
        :type bigram: list
        """
        for b in bigram:
            self.bigrams_ctr[b] += 1

    def update_trigram_counter(self, trigram):
        """Update the frequency counter
        :param trigram: List of trigrams
        :type trigram: list
        """
        for t in trigram:
            self.trigrams_ctr[t] += 1

    def update_quadgram_counter(self, quadgram):
        """Update the frequency counter
        :param quadgram: List of quadgrams
        :type quadgram: list
        """
        for q in quadgram:
            self.quadgrams_ctr[q] += 1

    def compute_ngrams(self, pickle_filename_unigrams='academic_unigrams.pkl', pickle_filename_bigrams='academic_bigrams.pkl', pickle_filename_trigrams='academic_trigrams.pkl', pickle_filename_quadgrams='academic_quadgrams.pkl'):
        """Compute the n-grams from the corpus
        :param pickle_filename_unigrams: File name for the academic unigrams counter pickle file
        :type pickle_filename_unigrams: str
        :param pickle_filename_bigrams: File name for the academic bigrams counter pickle file
        :type pickle_filename_bigrams: str
        :param pickle_filename_trigrams: File name for the academic trigrams counter pickle file
        :type pickle_filename_quadgrams: str
        :param pickle_filename_quadgrams: File name for the academic quadgrams counter pickle file
        :type pickle_filename_quadgrams: str
        """
        for root, _, files in os.walk(self.ACL_PATH):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if(os.path.isfile(file_path)):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        content = self.clean_content(content)

                        unigrams = self.get_ngram(content, 1)
                        self.update_unigram_counter(unigrams)

                        bigrams = self.get_ngram(content, 2)
                        self.update_bigram_counter(bigrams)

                        trigrams = self.get_ngram(content, 3)
                        self.update_trigram_counter(trigrams)

                        quadgrams = self.get_ngram(content, 4)
                        self.update_quadgram_counter(quadgrams)

        with open(os.path.join(self.SERIALIZE, pickle_filename_unigrams), 'wb') as f:
            pickle.dump(self.unigrams_ctr, f)

        with open(os.path.join(self.SERIALIZE, pickle_filename_bigrams), 'wb') as f:
            pickle.dump(self.bigrams_ctr, f)

        with open(os.path.join(self.SERIALIZE, pickle_filename_trigrams), 'wb') as f:
            pickle.dump(self.trigrams_ctr, f)

        with open(os.path.join(self.SERIALIZE, pickle_filename_quadgrams), 'wb') as f:
            pickle.dump(self.quadgrams_ctr, f)

    def load_ngram_ctrs(self, pickle_filename_unigrams='academic_unigrams.pkl', pickle_filename_bigrams='academic_bigrams.pkl', pickle_filename_trigrams='academic_trigrams.pkl', pickle_filename_quadgrams='academic_quadgrams.pkl'):
        """Loads the n-grams counters from the pickle files
        :param pickle_filename_bigrams: File name for the academic bigrams counter pickle file
        :type pickle_filename_bigrams: str
        :param pickle_filename_trigrams: File name for the academic trigrams counter pickle file
        :type pickle_filename_quadgrams: str
        :param pickle_filename_quadgrams: File name for the academic quadgrams counter pickle file
        :type pickle_filename_quadgrams: str
        """
        with open(os.path.join(self.SERIALIZE, pickle_filename_unigrams), 'rb') as f:
            self.unigrams_ctr = pickle.load(f)

        with open(os.path.join(self.SERIALIZE, pickle_filename_bigrams), 'rb') as f:
            self.bigrams_ctr = pickle.load(f)

        with open(os.path.join(self.SERIALIZE, pickle_filename_trigrams), 'rb') as f:
            self.trigrams_ctr = pickle.load(f)

        with open(os.path.join(self.SERIALIZE, pickle_filename_quadgrams), 'rb') as f:
            self.quadgrams_ctr = pickle.load(f)


def prep_academic_corpus(raw_academic_corpus, text_academic_corpus):
    """Extracts the text portion from the tar XML file of the ACL anthology corpus.
    :param raw_academic_corpus: base directory name of the tar file
    :type pickle_filename_bigrams: str
    :param pickle_filename_trigrams: File name for the output - text portion the XML files
    :type pickle_filename_quadgrams: str
    """
    for f in os.listdir(raw_academic_corpus):
        tar = tarfile.open(raw_academic_corpus+f, 'r:gz')
        for member in tar.getmembers():
            f = tar.extractfile(member)
            content = f.read().decode('utf-8')
            soup = BeautifulSoup(content, 'xml')
            new_file_name = text_academic_corpus + member.name[:-3] + 'txt'
            directory_path = os.path.dirname(new_file_name)
            try:
                os.makedirs(directory_path)
            except FileExistsError:
                pass
            with open(new_file_name, 'w') as f:
                for x in soup.findAll('bodyText'):
                    f.write(x.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes the n-gram distribution from academic corpus')
    parser.add_argument('--raw_academic_corpus', help='Path to raw academic corpus - ACL anthology')
    parser.add_argument('--text_academic_corpus', help='Path to text academic corpus', required=True)
    parser.add_argument('--serialize_output', help='Path to output picke objects', required=True)
    args = parser.parse_args()

    if(args.raw_academic_corpus):
        prep_academic_corpus(args.raw_academic_corpus, args.text_academic_corpus)

    academic = Academic(acl_path=args.text_academic_corpus, serialize_path=args.serialize_output)
    academic.compute_ngrams()
    # academic.load_ngram_ctrs()
