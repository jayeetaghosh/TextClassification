"""
Utility functions
@author jghosh@trace3.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
import tensorflow as tf
from keras.utils import np_utils

# Read Datasets
def read_data():
    impact_cols = ["PART_NO","MODEL_NO","ROW_NO"]
    impact = pd.read_csv("PartImpactData/impact.csv", names=impact_cols,
                         dtype={'MODEL_NO': 'category'})#(50961, 3)
    impact.drop('ROW_NO', axis=1, inplace=True)

    part_cols =["CDG_PART_NO","SRC_DB","PART_NAME","PART_DESCRIPTION","PART_IDENTIFIER","PIN","ATN",
                "PART_CLASS","AUTHORITY_SYSTEM","ENTERPRISE_IDENTIFIER","ENTERPRISE_TYPE",
                "ELECTROSTATIC_DISCH_SENS_IND","LIFE_LIMITED_INDICATOR","IS_TOP_LEVEL_PART","FEED_TS",
                "SOURCE_START_TS","CDW_START_TS","ROW_NO"]
    parts = pd.read_csv("PartImpactData/part.csv", names=part_cols,
                        dtype={'ELECTROSTATIC_DISCH_SENS_IND':'category',
                               'LIFE_LIMITED_INDICATOR':'category',
                               'IS_TOP_LEVEL_PART':'category'})#(202262, 18)
    parts.drop(['ROW_NO','SRC_DB', "PART_IDENTIFIER","PIN","ATN","PART_CLASS","AUTHORITY_SYSTEM",
                "ENTERPRISE_IDENTIFIER","ENTERPRISE_TYPE"], axis=1, inplace=True)
    # print(impact.apply(pd.Series.nunique))
    #print(parts.apply(pd.Series.nunique))

    # Merge impact dataset with parts
    dataset = pd.merge(parts, impact,  how='left', left_on=['CDG_PART_NO'], right_on = ['PART_NO'])

    # For now delete TS time stamps variables
    dataset.drop(['FEED_TS', 'SOURCE_START_TS','CDW_START_TS', 'PART_NO'], axis=1, inplace=True)
    # print(dataset.isnull().sum())

    # Handle missing values
    dataset['ELECTROSTATIC_DISCH_SENS_IND'] = dataset['ELECTROSTATIC_DISCH_SENS_IND'].replace(np.nan, "Missing")

    # Make sure cols are treated as categorical
    for col in ['ELECTROSTATIC_DISCH_SENS_IND', 'MODEL_NO']:
        dataset[col] = dataset[col].astype('category')

    # Handle target
    # from numpy import argmax
    # from keras.utils import to_categorical
    # encoded = to_categorical(dataset['MODEL_NO'])
    # print(encoded)
    # # invert encoding
    # inverted = argmax(encoded[0])
    # print(inverted)
    dataset['MODEL_NO'] = dataset['MODEL_NO'].replace(np.nan, "missing")
    # dataset['MODEL_NO'] = dataset['MODEL_NO'].astype('category')
    return dataset


# Convert Categorical to integer
def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category']))

    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding ' + feature)
    return df

# Merge two text columns
def merge_textcols (df):
    df['PART_NAME'] = df['PART_NAME'].replace(np.nan, "missing")
    df['PART_DESCRIPTION'] = df['PART_DESCRIPTION'].replace(np.nan, "missing")
    df['PART_TEXT'] = df["PART_NAME"].map(str) + ' ' + df["PART_DESCRIPTION"].map(str)

    # Remove duplicate words after merging; lot of time both columns contain same info
    df["PART_TEXT"] = df["PART_TEXT"].apply(lambda x: ' '.join(pd.unique(x.split())))

    # Handle missing and drop two original text cols
    # df['PART_TEXT'] = df['PART_TEXT'].replace(np.nan, "missing")
    df.drop(['PART_NAME', 'PART_DESCRIPTION'], axis=1, inplace=True)
    return df


#############################
# BOW features
#############################
def get_bow_features(df, num_row, MAX_BOW_FEA, keepNum, handleAbbr, ngram):
    num_rows = min(num_row, df.shape[0])

    # Initialize an empty list to hold the clean text
    clean_text = []

    # Loop over each text row, append to the list after cleaning
    for i in range(0, num_rows):
        if ((i + 1) % 1000 == 0):
            print("Row %d of %d\n" % (i + 1, num_rows))
        x = df['PART_TEXT'][i]
        word = text_to_words(x, keepNum, handleAbbr)
        clean_text.append(word)

    # vectorizer = CountVectorizer(max_features=MAX_BOW_FEA)
    vectorizer = CountVectorizer(analyzer="word", ngram_range=ngram ,
                                 stop_words = ["missing", "nan"], max_features=MAX_BOW_FEA)
    print(clean_text)
    # bow_transformer = vectorizer.fit_transform(df['PART_TEXT'])
    bow_transformer = vectorizer.fit_transform(clean_text)

    bow = bow_transformer.toarray()
    print(bow.shape)

    bow_df = pd.DataFrame(bow)
    # Get the vocab
    bow_features = vectorizer.get_feature_names()
    # print(bow_features)

    bow_df.columns = [bow_features]

    # Get stats for each words
    # Sum up the counts of each vocabulary word
    dist = np.sum(bow, axis=0)

    # Get word_count
    distdf = pd.DataFrame({'tag': bow_features, 'count': dist})
    distdf.to_csv("Output/word_count.csv", index=False)
    # bow_df.to_csv("Output/bow.csv", index=False)
    #
    #  for tag, count in zip(bow_features, dist):
    #     print(count, tag)

    return bow_df


def get_all_features(originaldf, bowdf):
    bowdf['CDG_PART_NO'] = originaldf['CDG_PART_NO']
    bowdf['ELECTROSTATIC_DISCH_SENS_IND'] = originaldf['ELECTROSTATIC_DISCH_SENS_IND']
    bowdf['LIFE_LIMITED_INDICATOR'] = originaldf['LIFE_LIMITED_INDICATOR']
    bowdf['IS_TOP_LEVEL_PART'] = originaldf['IS_TOP_LEVEL_PART']
    bowdf['MODEL_NO'] = originaldf['MODEL_NO']
    return bowdf


def text_to_words ( raw_text, keepNum, handleAbbr):
    # Function to convert a raw text to a string of words
    # Remove HTML
    nohtml_text = BeautifulSoup(raw_text).get_text()

    # Remove non-letters
    if(keepNum):
        # Remove non-letters and non-numbers
        letteronly_text = re.sub("[^a-zA-Z0-9]", " ", nohtml_text)
    else:
        letteronly_text = re.sub("[^a-zA-Z]", " ", nohtml_text)

    # Handle Abbreviation - its not working
    abbreviation_dict = {' assy ':' assembly ',
                         'inch':'inches',
                         ' instl ':' install ',
                         '.*\b(slat)\b.*': 'Jayeeta',
                         }
    abbreviation_re = re.compile('(%s)' % '|'.join(abbreviation_dict.keys()))
    def expand_contractions(s, contractions_dict=abbreviation_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return abbreviation_re.sub(replace, s)

    if(handleAbbr):
        abbr_removed = expand_contractions(letteronly_text)
    else:
        abbr_removed = letteronly_text



    # Convert to lower case and split into individual words
    words = abbr_removed.lower().split()

    # stemming of words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    # print(stemmed[:100])

    # Lemmatizer
    # import nltk
    # lemma = nltk..wordnet.WordNetLemmatizer()
    # lemma.lemmatize('article')
    # 'article'
    # lemma..lemmatize('leaves')
    # 'leaf'

    # Convert the stop words into a set
    stops = set(stopwords.words("english"))

    # Remove stop words
    good_words = [w for w in words if not w in stops]

    # Join the good words into a string
    return (" ".join(good_words))

# # 1. INSTANTIATE
# enc = preprocessing.OneHotEncoder()
#
# # 2. FIT
# enc.fit(X_2)
#
# # 3. Transform
# onehotlabels = enc.transform(X_2).toarray()
# onehotlabels.shape
def preprocess_labels(labels, encoder=None, categorical=True):
    """Encode labels with values among 0 and `n-classes-1`"""
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
        le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print("Mapping",le_name_mapping)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

# import re
# abbreviation_dict = {'assy':'assembly',
#                      'inch':'inches',
#                       'instl':'install'}
# abbreviation_re = re.compile('(%s)' % '|'.join(abbreviation_dict.keys()))
# def expand_contractions(s, contractions_dict=abbreviation_dict):
#     def replace(match):
#         return contractions_dict[match.group(0)]
#     return abbreviation_re.sub(replace, s)
# expand_contractions('Now lets assy')

# Handle abbreviations
# >>> import re
# >>> contractions_dict = {
# ...     'didn\'t': 'did not',
# ...     'don\'t': 'do not',
# ... }
# >>> contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
# >>> def expand_contractions(s, contractions_dict=contractions_dict):
# ...     def replace(match):
# ...         return contractions_dict[match.group(0)]
# ...     return contractions_re.sub(replace, s)
# ...
# >>> expand_contractions('You don\'t need a library')
# 'You do not need a library'
print("Done!")

