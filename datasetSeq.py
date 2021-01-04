#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time
import os
import joblib
import numpy as np
import scipy
import pandas as pd
import math
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as bs
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# In[2]:

NUM_SAMPLES = None
NUM_TAGS = 4000
NUM_FEATURES = 1000

class SOFData(object):

    data_root = './data/archive'

    def __init__(self, root=None, N=-1):

        self.root = root        
        if not self.root:
            if os.path.exists(SOFData.data_root):
                self.root = SOFData.data_root
            else:
                print("Path {} does not exist".format(SOFData.data_root))
        
        self.tr_data = {'X': None, 'Y': None}
        self.te_data = {'X': None, 'Y': None}
        self.cl_data = None
        self.N = N
         
               
    def get_data(self, train=False, test=False, classes=True):      
        if train:
            print("Loading TRAINING dataset ...", end=" ")
            self.tr_data['X'] = joblib.load(f"{self.root}/Xtrain.pkl") 
            self.tr_data['Y'] = joblib.load(f"{self.root}/Ytrain.pkl")
            print("done.")
        if test:
            print("Loading TESTING dataset ...", end=" ")
            self.te_data['X'] = joblib.load(f"{self.root}/Xtest.pkl")
            self.te_data['Y'] = joblib.load(f"{self.root}/Ytest.pkl")
            print("done.")
        if classes:
            print("Loading CLASSES ...", end=" ")
            self.cl_data = joblib.load(f"{self.root}/Yclasses.pkl")
            print("done.") 

    def load_csv(self, num_rows=None):
        self.q_df = pd.read_csv(os.path.join(self.root, "Questions.csv"), 
                            encoding="ISO-8859-1", nrows=num_rows)

        #self.a_df = pd.read_csv(os.path.join(self.root, "Answers.csv"), 
        #                    encoding="ISO-8859-1", nrows=num_rows)

        self.t_df = pd.read_csv(os.path.join(self.root, "Tags.csv"), 
                            encoding="ISO-8859-1", nrows=num_rows)
        ##################################################
        self.q_df["Body"] = self.q_df["Body"].astype(str)
        self.q_df["Title"] = self.q_df["Title"].astype(str)
        self.t_df["Tag"] = self.t_df["Tag"].astype(str)
        ##################################################

    def see_df(df, h=False, t=False, ns=5):
        if h:
            print(df.head(ns))
        if t:
            print(df.tail(ns))

    def prepTags(self):
        # Unique tag dictionary
        self.unq_tags = self.t_df["Tag"].unique()
        self.Tag = {}
        self.IdToTag = {}
        for i, tag in enumerate(self.unq_tags):
            self.Tag[tag] = i + 1
            self.IdToTag[i + 1] = tag
        self.tech_tags = ['html5', 'node.js', 'x86', 'f#', 'asp.net', 
             'css3', '.net', 'unity3d', 'd3.js', '#', 
             'utf-8', '+', 'objective-c', 'c++', 
             'scikit-learn', '3d', 'c#', 'neo4j']
        #print(len(self.unq_tags), len(self.Tag.keys()))

    def unique_tags(self):
        self.unq_tag_df = self.t_df.copy()

        print("Preparing unique tags: ...")
        self.unq_tag_df["TagId"] = self.unq_tag_df["Tag"]\
                                   .progress_apply(lambda tag: self.Tag[tag])
        print()

        #print(self.unq_tag_df.head())
        #print(len(self.unq_tag_df["Tag"].unique()))
        #print(self.unq_tag_df[self.unq_tag_df["Tag"] == "c++"])

    def tag_frequency(self):
        self.tag_freq = self.unq_tag_df['Tag']\
                        .value_counts(sort=True)\
                        .rename_axis('Tag')\
                        .reset_index(name='Frequency')
        self.tag_freq["TagId"] = self.tag_freq["Tag"].progress_apply(lambda tag: self.Tag[tag])
        #print(self.tag_freq.head())          
       
    def plot_tag_freq(self):
        # In[23]:
        self.tag_freq.head(20).plot.bar(x='Tag', y='Frequency')

        # In[24]:
        self.tag_freq.tail(20).plot.bar(x='Tag', y='Frequency')

    def question_tags(self):
        # In[35]:

        self.Q_tags = self.unq_tag_df.groupby("Id")["Tag"].progress_apply(lambda tags: list(tags))

        # In[37]:       
        self.Q_tags = self.Q_tags.rename_axis('Id').reset_index(name='Tag')


        # In[38]:
        #print(self.Q_tags.head())


        # In[39]:
        self.Q_tagIds = self.unq_tag_df.groupby("Id")["TagId"].progress_apply(lambda tagIds: list(tagIds)) 


        # In[40]:
        self.Q_tagIds = self.Q_tagIds.rename_axis('Id').reset_index(name='TagId')


        # In[41]:
        self.Q_tagIds["Tag"] = self.Q_tags["Tag"]


        # In[42]:
        #print(self.Q_tagIds.head(30))

    
    def body_processor(self):
        nltk.download("punkt")
        self.q_df["Body"] = self.q_df["Body"].str.lower()
        self.removeTags("Body")
        self.text_tokenizer("Body")
        self.init_multi_word_tokenizer()
        self.multi_word_tokenizer(cname="Body_tokenized")
        self.filter_stop_words(cname="Body_tokenized")
    
    def title_processor(self):
        self.q_df["Title"] = self.q_df["Title"].str.lower()
        self.removeTags("Title")
        self.text_tokenizer("Title")
        self.init_multi_word_tokenizer()
        self.multi_word_tokenizer(cname="Title_tokenized")
        self.filter_stop_words(cname="Title_tokenized")
        
    def removeTags(self, cname):
        print("Removing html tags: ...")
        self.q_df[cname] = self.q_df[cname].progress_apply(lambda text: bs(text, "lxml").text)
        print()
        
    def text_tokens(self, text):
        return [w for w in nltk.word_tokenize(text) if w.isalpha() or w in self.tech_tags]
    
    def text_tokenizer(self, cname):
        dst_cname = cname + "_tokenized"
        src_cname = cname
        print(f"Tokenizing {src_cname}: ...")
        self.q_df[dst_cname] = self.q_df[src_cname].progress_apply(lambda txt: self.text_tokens(txt))
        print()
                
    def init_multi_word_tokenizer(self):
        self.mw_tokenizer = nltk.MWETokenizer(separator="")
        self.mw_tokenizer.add_mwe(("c", "#"))
        self.mw_tokenizer.add_mwe(("c", "+", "+"))
        self.mw_tokenizer.add_mwe(("f", "#"))
    
    def multi_word_tokens(self, tokens):
        return [tkn for tkn in self.mw_tokenizer.tokenize(tokens)]
    
    def multi_word_tokenizer(self, cname):
        print(f"Tokenizing multi-word {cname}: ...")
        self.q_df[cname] = self.q_df[cname].progress_apply(lambda tokens: self.multi_word_tokens(tokens))
        print()       
 
    def remove_stopWords(self, words):
        return [w for w in words if not w in self.stopWords]
    
    def filter_stop_words(self, cname):
        # Stop-words
        nltk.download("stopwords")
        self.stopWords = set(stopwords.words("english"))
        
        print(f"Removing stop-words in {cname}: ...")
        self.q_df[cname] = self.q_df[cname].progress_apply(lambda words: self.remove_stopWords(words))
        print()       
 
    def joinQuestionToTags(self):
        self.q_tags_df = pd.merge(self.q_df, self.Q_tagIds, left_on='Id', right_on='Id')
        
    def prepQuestionsTagsDF(self):
        cols = ["OwnerUserId", "CreationDate", "ClosedDate", "Body", "Title"]
        for c in cols:
            del self.q_tags_df[c]
            
    def saveDataFrameToCsv(self, df, fname):
        df.to_csv(fname, index=False)
        
    def getPopularTags(self, top=1):
        self.topNumTags = top
        self.pop_tag_freq = self.tag_freq.head(self.topNumTags)
        
    def keepPopularTags(self, tagList):
        new_tags = []
        for tag in tagList:
            if tag in list(self.pop_tag_freq["Tag"]):
                new_tags.append(tag)
        return new_tags

    def keepPopularTagIds(self, tagList):
        return [self.Tag[tag] for tag in tagList]
    
    def prepPopularTagsDataFrame(self):
        self.pop_tags_df = self.q_tags_df.copy()
        print(f"Keeping samples with popular tags only: ...")
        self.pop_tags_df["Tag"] = self.q_tags_df["Tag"].progress_apply(lambda tagList: self.keepPopularTags(tagList))
        self.pop_tags_df["TagId"] = self.pop_tags_df["Tag"].progress_apply(lambda tagList: self.keepPopularTagIds(tagList))
        self.pop_tags_df["NumTags"] = self.pop_tags_df["Tag"].progress_apply(lambda tagList: len(tagList))
        print()
        self.pop_tags_df.drop(self.pop_tags_df[self.pop_tags_df['NumTags'] == 0].index, inplace = True)
        
        print("Dropped {} rows with all unpopular tags.".format(len(self.q_tags_df) - len(self.pop_tags_df)))
        print("{} rows with popular tags.".format(len(self.pop_tags_df)))

    def prepFeatures(self, num_features=100):
        # Limit features for title and body
        self.num_features = num_features

        self.T_vectorizer = TFIDV(tokenizer = lambda s: s, 
                            lowercase = False, 
                            max_features = num_features)
        self.B_vectorizer = TFIDV(tokenizer = lambda s: s, 
                            lowercase = False, 
                            max_features = num_features)
        
        self.title_feats = self.T_vectorizer.fit_transform(self.pop_tags_df["Title_tokenized"])
        self.body_feats = self.B_vectorizer.fit_transform(self.pop_tags_df["Body_tokenized"])
        
        #print(pd.DataFrame(self.title_feats[:11].toarray(), columns=self.T_vectorizer.get_feature_names())\
        #        .iloc[10]\
        #        .sort_values(ascending=False)\
        #        .where(lambda v: v > 0).dropna().head(10))
        #print(pd.DataFrame(self.body_feats[:11].toarray(), columns=self.B_vectorizer.get_feature_names()).iloc[10].sort_values(ascending=False).where(lambda v: v > 0).dropna().head(10))
        
    def unqTags(self, df):
        print(f"Checking unique tags in DataFrame: ...")
        unqTags = list()
        for tagList in tqdm(df["Tag"]):
            for tag in tagList: 
                if tag not in unqTags:
                    unqTags.append(tag)
        print(f"Number of unique tags {len(unqTags)}")
        
    def prepDataset(self, savePkl=(False, "")):
        # Give title features more importance
        self.title_feats = self.title_feats * 2
        self.X = scipy.sparse.hstack([self.title_feats, self.body_feats])
        
        self.multiClassLabelCreator = MultiLabelBinarizer()
        self.label_df = self.pop_tags_df[["Tag"]]
        self.Y = self.multiClassLabelCreator.fit_transform(self.label_df["Tag"])
        
        self.tr_data['X'], self.te_data['X'], self.tr_data['Y'], self.te_data['Y'] = train_test_split(self.X, 
                                                                                                      self.Y, 
                                                                                                      test_size=0.2, 
                                                                                                      random_state = 1)
        self.cl_data = self.multiClassLabelCreator.classes_
        
        if savePkl[0]:
            # Save dataset files
            joblib.dump(self.tr_data['X'], f"{savePkl[1]}/Xtrain.pkl")
            joblib.dump(self.te_data['X'], f"{savePkl[1]}/Xtest.pkl")
            joblib.dump(self.tr_data['Y'], f"{savePkl[1]}/Ytrain.pkl")
            joblib.dump(self.te_data['Y'], f"{savePkl[1]}/Ytest.pkl")
            joblib.dump(self.multiClassLabelCreator.classes_, f"{savePkl[1]}/Yclasses.pkl") 

# In[3]:


def main():
    try:
        # Number of ___
        numRows = NUM_SAMPLES
        topTags = NUM_TAGS
        numFeatures = NUM_FEATURES

        # Path to save dataset pkl files
        data_dir = "./data"

        dataset = SOFData()

        dataset.load_csv(num_rows=numRows)

        #print(dataset.t_df.head())

        # Peformance evaluation
        preproc_st = time.time()

        dataset.prepTags()

        #print(dataset.tech_tags)

        dataset.unique_tags()

        dataset.tag_frequency()

        dataset.question_tags()

        dataset.body_processor()

        #print(dataset.q_df.head())

        dataset.title_processor()

        #print(dataset.q_df.shape)
        #print(dataset.q_df.head())

        dataset.joinQuestionToTags()

        #print(dataset.q_tags_df.shape)
        #print(dataset.q_tags_df.head())

        dataset.prepQuestionsTagsDF()

        #print(dataset.q_tags_df.shape)
        #print(dataset.q_tags_df.head())

        dataset.saveDataFrameToCsv(dataset.q_tags_df, "QuestionsTags.csv")

        dataset.getPopularTags(top=topTags)

        #print(dataset.pop_tag_freq.head())

        dataset.prepPopularTagsDataFrame()

        #print(dataset.pop_tags_df.head())

        #print(dataset.q_tags_df)

        dataset.saveDataFrameToCsv(dataset.pop_tags_df, "PopularQstnTags.csv")

        dataset.unqTags(dataset.pop_tags_df)

        dataset.prepFeatures(num_features = numFeatures)
        # Peformance evaluation
        preproc_et = time.time() - preproc_st
        print(f"Sequential time: {preproc_et} seconds.")
        #dataset.prepDataset(savePkl=(True, data_dir))

        print(f"Number of samples: {NUM_SAMPLES}")
        print(f"Number of tags: {NUM_TAGS}")
        print(f"Number of features: {NUM_FEATURES}") 
    except KeyboardInterrupt:
        # Peformance evaluation
        preproc_et = time.time() - preproc_st
        print(f"Sequential time: {preproc_et} seconds.")


if __name__ == "__main__":
    main()

