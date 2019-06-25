import os, re, pickle
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix

class Keyword:

    def __init__(self, acl_path, serialize_path):
        self.ACL_PATH = acl_path
        self.SERIALIZE = serialize_path
        self.corpus = list()
        self.file_order = list()
        self.tfidf_keywords_list = list()
        self.structure_keywords_list = list()

    def clean_content(self, content, remove_stopwords=False, lemmatize_words=True, remove_num=True):
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

        if(not lemmatize_words and not remove_stopwords):
            text = word_tokens

        if(remove_num): # Remove numbers
            text = [word for word in word_tokens if not word.isdigit()]
        content = " ".join(text)
        return content

    def build_academic_corpus(self, pickle_filename_corpus='corpus.pkl', pickle_filename_file_order='file_order.pkl'):
        """Compile the corpus into required format - list of strings
        :param pickle_filename_corpus: File name for the corpus pickle file - default 'corpus.pkl'
        :type pickle_filename_corpus: str
        :param pickle_filename_file_order: File name for the file order pickle file - default 'file_order.pkl'
        :type pickle_filename_file_order: str
        """
        for root, _, files in os.walk(self.ACL_PATH):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if(os.path.isfile(file_path)):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        content = self.clean_content(content, lemmatize_words=False)
                        self.file_order.append(file_path)
                        self.corpus.append(content)
        
        with open(os.path.join(self.SERIALIZE, pickle_filename_corpus), 'wb') as f:
            pickle.dump(self.corpus, f)

        with open(os.path.join(self.SERIALIZE, pickle_filename_file_order), 'wb') as f:
            pickle.dump(self.file_order, f)
        
    def load_corpus_and_file_order(self, pickle_filename_corpus='corpus.pkl', pickle_filename_file_order='file_order.pkl'):
        """Load the corpus and file-order data structure from the pickle file
        :param pickle_filename_corpus: File name for the corpus pickle file - default 'corpus.pkl'
        :type pickle_filename_corpus: str
        :param pickle_filename_file_order: File name for the file order pickle file - default 'file_order.pkl'
        :type pickle_filename_file_order: str
        """
        with open(os.path.join(self.SERIALIZE, pickle_filename_corpus), 'rb') as f:
            self.corpus = pickle.load(f)

        with open(os.path.join(self.SERIALIZE, pickle_filename_file_order), 'rb') as f:
            self.file_order = pickle.load(f)

    def compute_tfidf(self, top_n=3015, pickle_filename_keywords='tf-idf_keywords(1,4).pkl', ngram_range=(1, 4)):
        """ Compute the TF-IDF score to estimate the key phrases in the document. The number of entries in the COCA list is 3015.
        :param top_n: Number of keyphrases to be extracted. Default = 3015 (the number of entries in the COCA list).
        :type top_n:  int
        :param pickle_filename_keywords: File name for the keywords pickle file - default 'tf-idf_keywords(1,4).pkl'
        :type pickle_filename_keywords: str
        :param ngram_range: Range of ngrams to consider. Hyper parameter of TfidfVectorizer
        :type ngram_range: tuple
        """
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=top_n)
        X = vectorizer.fit_transform(self.corpus)
        self.tfidf_keywords_list = vectorizer.get_feature_names()

        with open(os.path.join(self.SERIALIZE, pickle_filename_keywords), 'wb') as f:
            pickle.dump(self.tfidf_keywords_list, f)


    def load_tfidf_keywords(self, pickle_filename_keywords='tf-idf_keywords(1,4).pkl'):
        """ Load the TF-IDF keywords from the pickle file
        :param pickle_filename_keywords: File name for the keywords pickle file - default 'tf-idf_keywords(1,4).pkl'
        :type pickle_filename_keywords: str
        """
        with open(os.path.join(self.SERIALIZE, pickle_filename_keywords), 'rb') as f:
            self.tfidf_keywords_list = pickle.load(f)


def prep_academic_corpus(raw_academic_corpus, text_academic_corpus):
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
    parser = argparse.ArgumentParser(description='Computes the key phrases from academic corpus')

    parser.add_argument('--raw_academic_corpus', help='Path to raw academic corpus - ACL anthology')
    parser.add_argument('--text_academic_corpus', help='Path to text academic corpus', required=True)

    parser.add_argument('--serialize_output', help='Path to output pickle objects', required=True)
    args = parser.parse_args()

    if(args.raw_academic_corpus):
        prep_academic_corpus(args.raw_academic_corpus, args.text_academic_corpus)

    model = Keyword(acl_path=args.text_academic_corpus, serialize_path=args.serialize_output)
    model.build_academic_corpus()
    model.load_corpus_and_file_order()
    model.compute_tfidf()
