import os, pickle
import argparse
import pandas as pd

from academic_ngrams import Academic
from non_academic_ngrams import Non_Academic

class Frequency:

    def __init__(self, academic, non_academic, key_phrases, sheet, academic_file_name=None, non_academic_file_name=None):
        self.academic = academic
        self.non_academic = non_academic
        self.key_phrases = list(set(key_phrases))
        self.sheet = sheet
        self.academic_file_name = academic_file_name
        self.non_academic_file_name = non_academic_file_name

    def get_freq(self, phrase):
        word_list = tuple(phrase.split())
        if(len(word_list) == 1):
            return (self.academic.unigrams_ctr[word_list], self.non_academic.unigrams_ctr[word_list])
        elif(len(word_list) == 2):
            return (self.academic.bigrams_ctr[word_list], self.non_academic.bigrams_ctr[word_list])
        elif(len(word_list) == 3):
            return (self.academic.trigrams_ctr[word_list], self.non_academic.trigrams_ctr[word_list])
        elif(len(word_list) == 4):
            return (self.academic.quadgrams_ctr[word_list], self.non_academic.quadgrams_ctr[word_list])
        else:
            return (0, 0) # Word of vocabulary multi word phrases

    def compile_doc(self):
        academic_phrases = list()
        non_academic_phrases = list()
        academic_freq = list()
        non_academic_freq = list()
        ratio_list = list()
        print(self.academic_file_name, self.non_academic_file_name)
        for phrase in self.key_phrases:
            freq = self.get_freq(phrase) # (academic_freq, non_academic_freq)
            acad_freq = freq[0]
            non_acad_freq = freq[1]
            try:
                if(self.academic_file_name):
                    ratio = float(acad_freq)/non_acad_freq
                    if(ratio >= 1.5):
                        academic_phrases.append(phrase)
                        ratio_list.append(ratio)
                        academic_freq.append(acad_freq)
                        non_academic_freq.append(non_acad_freq)
                elif(self.non_academic_file_name):
                    ratio = float(non_acad_freq)/acad_freq
                    if(ratio >= 1.5):
                        non_academic_phrases.append(phrase)
                        ratio_list.append(ratio)
                        academic_freq.append(acad_freq)
                        non_academic_freq.append(non_acad_freq)
            except:
                continue # To deal with the case where the phrase has 0 frequency in non-academic corpora
        if(self.academic_file_name):
            d = {'phrase': academic_phrases, 'academic_freq': academic_freq, 'non_academic_freq': non_academic_freq, 'ratio': ratio_list}
            df = pd.DataFrame(d)
            df.to_excel(self.academic_file_name, sheet_name=self.sheet, index=False)
        elif(self.non_academic_file_name):
            d = {'phrase': non_academic_phrases, 'academic_freq': academic_freq, 'non_academic_freq': non_academic_freq, 'ratio': ratio_list}
            df = pd.DataFrame(d)
            df.to_excel(self.non_academic_file_name, sheet_name=self.sheet, index=False)


def load_keywords(file_path):
    with open(file_path, 'rb') as f:
        keywords = pickle.load(f)
    return keywords


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes the frequency distribution from corpus')
    parser.add_argument('--non_academic', help='Mention if non-academic list is to be generated', action='store_true')
    parser.add_argument('--raw_academic_corpus', help='Path to raw academic corpus - ACL anthology')
    parser.add_argument('--text_academic_corpus', help='Path to text academic corpus', required=True)
    parser.add_argument('--text_non_academic_corpus', help='Path to raw non-academic corpus', required=True)
    parser.add_argument('--serialize_output', help='Path to output picke objects', required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tfidf_keywords', help='Path to input TF-IDF keywords list')
    group.add_argument('--pos_dependency_keywords', help='Path to input POS + Dependency keywords list')

    ACADEMIC_FILE_NAME = 'academic_keyphrases.xlsx'
    NON_ACADEMIC_FILE_NAME = 'non_academic_keyphrases.xlsx'

    args = parser.parse_args()

    if(args.raw_academic_corpus):
        prep_academic_corpus(args.raw_academic_corpus, args.text_academic_corpus)


    academic = Academic(acl_path=args.text_academic_corpus, serialize_path=args.serialize_output)
    academic.load_ngram_ctrs()

    non_academic = Non_Academic(amazon_review=args.text_non_academic_corpus, serialize_path=args.serialize_output)
    non_academic.load_ngram_ctrs()

    if(args.tfidf_keywords):
        tfidf_keywords = load_keywords(args.tfidf_keywords)
        if(not args.non_academic):
            freq = Frequency(academic, non_academic, tfidf_keywords, 'tf-idf', academic_file_name=ACADEMIC_FILE_NAME)
        else:
            freq = Frequency(academic, non_academic, tfidf_keywords, 'tf-idf', non_academic_file_name=NON_ACADEMIC_FILE_NAME)
        freq.compile_doc()
    elif(args.pos_dependency_keywords):
        pos_dependency_keywords = load_keywords(args.pos_dependency_keywords)
        if(not args.non_academic):
            freq = Frequency(academic, non_academic, pos_dependency_keywords, 'pos-dependency', academic_file_name=ACADEMIC_FILE_NAME) 
        else:
            freq = Frequency(academic, non_academic, pos_dependency_keywords, 'pos-dependency', non_academic_file_name=NON_ACADEMIC_FILE_NAME)
        freq.compile_doc()
