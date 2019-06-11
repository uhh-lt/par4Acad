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
from scipy.sparse import coo_matrix

class Keyword:

    def __init__(self, acl_path, serialize_path):
        self.ACL_PATH = acl_path
        self.SERIALIZE = serialize_path
        self.corpus = list()
        self.file_order = list()
        self.tfidf_keywords_list = list()
        self.structure_keywords_list = list()

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
                        content = self.clean_content(content)
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

    def compute_tfidf(self, top_n=10, pickle_filename_keywords='tf-idf_keywords(1,4).pkl', max_df=0.8, max_features=10000, ngram_range=(1, 4), smooth_idf=True, use_idf=True):
        """ Compute the TF-IDF score to estimate the key phrases in the document. The number of entries in the COCA list is 3015.
        :param top_n: Number of keyphrases to be extracted
        :type top_n:  int
        :param pickle_filename_keywords: File name for the keywords pickle file - default 'tf-idf_keywords(1,4).pkl'
        :type pickle_filename_keywords: str
        :param max_df: Hyper parameter for CountVectorizer
        :type max_df: float
        :param max_features: Hyper paramter for CountVectorizer
        :type max_features: int
        :param ngram_range: Range of ngrams to consider. Hyper parameter of CountVectorizer
        :type ngram_range: tuple
        :param smooth_idf: Hyper parameter of TfidfTransformer
        :type smooth_idf: bool
        :param use_idf: Hyper parameter of TfidfTransformer
        :type use_idf: bool
        """
        cv = CountVectorizer(max_df, max_features, ngram_range)
        X = cv.fit_transform(self.corpus)

        def sort_coo(coo_matrix):
            tuples = zip(coo_matrix.col, coo_matrix.data)
            return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

        def extract_topn_from_vector(feature_names, sorted_items, topn):
            sorted_items = sorted_items[:topn]
            score_vals = []
            feature_vals = []
            
            for idx, score in sorted_items:
                score_vals.append(round(score, 3))
                feature_vals.append(feature_names[idx])
            
            results = {}
            for idx in range(len(feature_vals)):
                results[feature_vals[idx]] = score_vals[idx]
            return results


        tfidf_transformer = TfidfTransformer(smooth_idf=smooth_idf, use_idf=use_idf)
        tfidf_transformer.fit(X)

        feature_names = cv.get_feature_names()

        for i in range(len(self.corpus)):
            file_path = self.file_order[i]
            tfidf_vector = tfidf_transformer.transform(cv.transform([self.corpus[i]]))
            sorted_items = sort_coo(tfidf_vector.tocoo())
            keywords = extract_topn_from_vector(feature_names, sorted_items, top_n)
            self.tfidf_keywords_list.append((file_path, keywords))


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
    model.load_tfidf_keywords()
