from logging import log
import pickle
import re
import sys, os, json
import numpy as np
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
from collections import Counter
import pandas as pd
from tqdm import tqdm


def get_ids_dict_dfs(doc_name, index_reader):
    vector_dict = dict()
    words_df_dict = dict()
    docids = list(pd.read_json(doc_name, lines=True)['id'])
    for id in tqdm(docids):
        vec = index_reader.get_document_vector(str(id))
        vector_dict[str(id)] = vec
        for word in vec.keys():
            if word not in words_df_dict.keys():
                df, cf = index_reader.get_term_counts(word, analyzer=None)
                words_df_dict[word] = df
    return (docids, vector_dict, words_df_dict)

def judge_all_config(feature_dict, l):
    for item in l:
        if not feature_dict[item]:
            return False
    return True


class Ranker(object):
    '''
    The base class for ranking functions. Specific ranking functions should
    extend the score() function, which returns the relevance of a particular 
    document for a given query.
    '''
    
    
    def __init__(self, index_reader, vectors, words_df):
        self.index_reader = index_reader
        self.doc_average_len = index_reader.stats()['total_terms'] / index_reader.stats()['documents']
        self.N = self.index_reader.stats()['documents']
        self.vector_dict = vectors
        self.df_dict = words_df

        
    def score(self, query, doc_id):        
        '''
        Returns the score for how relevant this document is to the provided query.
        Query is a tokenized list of query terms and doc_id is the identifier
        of the document in the index should be scored for this query.
        '''
        
        rank_score = 0
        return rank_score

    
class DemoRanker(Ranker):
    
    def __init__(self, index_reader, vectors, words_df, index_fname='./index'):
        super().__init__(index_reader=index_reader, vectors=vectors, words_df=words_df)
        self.searcher = SimpleSearcher(index_fname)


    def score(self, query, doc_id, dict_feature_extraction=None, enhanced=False, big_mtp=0.4, small_mtp=0.1, k1=1.2, b=0.75, k3=1.2):

        feature_dict = json.loads(self.searcher.doc(doc_id).raw())
        feature_exterior = 'exterior_color'
        feature_configuration = ['heated_seats', 'heated_steering_wheel', 'nav_sys', \
            'remote_start', 'carplay', 'bluetooth', 'brake_assist', 'blind_spot_monitor']
        feature_price = 'price_level'
        feature_ftype = 'fuel_type_simple'
        feature_power = ['emission', 'transmission_simple']
        extent_words = {
            'cheap': ['small', 'low', 'cheap', 'less', 'economic', 'cost-effective'],
            'moderate': ['middle', 'moderate', 'fair', 'average', 'family'],
            'expensive': ['big', 'large', 'great', 'more', 'fast', 'strong', 'high', 'expensive', 'luxury', 'performance', 'good', 'racecar']
        }
        config_cat = {
            'safety': ['brake_assist', 'blind_spot_monitor'],
            'comfort': ['heated_seats', 'heated_steering_wheel', 'remote_start'],
            'multimedia': ['nav_sys', 'carplay', 'bluetooth']
        }
        drivetrain = {'two wheel drive': ['RWD', 'FWD'], 'four wheel drive': ['AWD'], 'all wheel drive': ['AWD']}

        enhance_score = 0
        if enhanced:
            if feature_dict['make'].lower() in query:
                enhance_score += big_mtp

            for k, v in drivetrain.items():
                if k in ' '.join(query) and feature_dict['drive_train_simple'] in v:
                    enhance_score += big_mtp

            if len(set(feature_dict[feature_ftype].lower().split()).intersection(set(query))) > 0:
                enhance_score += big_mtp

            # add this to argument
            # dict_feature_extraction = {
            # "blue car": "exterior",
            # "drive car": "exterior",
            # "safety configurations": "configuration"
            # }
            
            # foo(' '.join(query))
            for k, v in dict_feature_extraction.items():
                if k == 'drive car':
                    continue

                if v == 'configuration':
                    for fp in feature_power:
                        enhance_score += small_mtp * int(feature_dict[fp])
                        for kk, vv in config_cat.items():
                            if kk in query and judge_all_config(feature_dict, vv):
                                enhance_score += big_mtp

                if v == 'exterior':
                    if len(set(feature_dict[feature_exterior].lower().split()).intersection(k.lower().split())) > 0:
                        enhance_score += big_mtp
                
                if v == 'price':
                    for kk, vv in extent_words.items():
                        if len(set(k.split()).intersection(set(vv))) > 0:
                            level = kk
                        else:
                            level = ''
                    if level == feature_dict[feature_price]:
                        enhance_score += big_mtp
                
                if v == 'power':
                    c_power_thres = [2.0, 5]
                    e_power_thres = [4.0, 8]

                    for kk, vv in extent_words.items():
                        if len(set(k.split()).intersection(set(vv))) > 0:
                            level = kk
                        else:
                            level = ''

                    for i in range(len(feature_power)):
                        val = feature_dict[feature_power[i]]
                        if level == 'cheap' and val <= c_power_thres[i]:
                            enhance_score += small_mtp
                        if level == 'expensive' and val > e_power_thres[i]:
                            enhance_score += small_mtp
                        if level == 'moderate' and val > c_power_thres[i] and val <= e_power_thres[i]:
                            enhance_score += small_mtp

        # convert feature to long string
        doc_str_long = ''
        for k, v in feature_dict.items():
            if k in feature_configuration:
                doc_str_long += k if v else ('no' + k)
            else:
                doc_str_long += str(v)
            doc_str_long += ' '
        vec_long = dict(Counter(doc_str_long.lower().split()))
        rank_score = 0
        vec = self.vector_dict[doc_id]
        # union two vec
        vec.update(vec_long)
        self.df_dict.update(vec_long)
        # print(vec)

        dl = sum(cnt for cnt in vec.values() if cnt != None)
        for q in query:
            if q in vec.keys():
                df = self.df_dict[q]
                if df == 0:
                    continue
                tf = vec[q]
                qtf = query.count(q)
                rank_score += np.log((self.N - df + 0.5)/(df + 0.5)) * \
                    (tf * (k1 + 1) / (k1 *(1 - b + b * dl / self.doc_average_len) + tf)) \
                        * (qtf * (k3 + 1)/(k3 + qtf))                
        
        return rank_score + enhance_score

if __name__ == "__main__":
    index_fname = "./index"
    doc_fname = "./data_dropped/data_dropped.json"
    index_reader = IndexReader(index_fname)  # Reading the indexes

    docids, vectors, words_df = get_ids_dict_dfs(doc_fname, index_reader)


    # Print some basic stats
    print("Loaded dataset with the following statistics: " + str(index_reader.stats()))
    
    print("Initializing Ranker")
    # Choose which ranker class you want to use
    ranker = DemoRanker(index_reader, vectors, words_df)

    print("Tesing Ranker!")
    sample_query = "I want a red Ford car with half price"  # sample Query to check
    for i in range(4):
        print(f"score for {i} is {ranker.score(sample_query.lower().split(' '), str(i), enhanced=True)}")
