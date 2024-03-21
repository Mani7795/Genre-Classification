import sqlite3
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import argparse
parser = argparse.ArgumentParser(description='Indexing Script.')
parser.add_argument('--raw-data',type=str,required=True, help='Path to the input text file.')
args = parser.parse_args()
connec = sqlite3.connect(args.raw_data)
df = pd.read_sql("Select tvmaze_id, description as text, showname  from tvmaze where description is not NULL group by text", connec)
import re
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def contains_link(text):
    return bool(re.search(url_pattern, text))
df1 = pd.DataFrame()
# Apply the function to each row in the DataFrame
df1 = (df[~df['text'].apply(contains_link)])
def remove_short_words(text):
    words = text.split()  # Split the text into words
    words = [word for word in words if len(word) >= 3]  # Remove words with length less than 2
    return ' '.join(words)
df1['text'] = df1['text'].apply(remove_short_words)


df1['text'] = df1['text'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
df1['text'] = df1['text'].str.lower()
custom_stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't",
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't",
    'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't",
    'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't",
    'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've",
    'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
    'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
    'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
    'yourselves']
def remove_custom_stopwords(text):
    words = text.split()  # Split the text into words
    words = [word for word in words if word.lower() not in custom_stopwords]
    return ' '.join(words)
df1['text'] = df1['text'].apply(remove_custom_stopwords)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word, wordnet.VERB) for word in tokens]
    return ' '.join(lemmatized_tokens)

#df = pandas.read_sql("select tvmaze_id, description from tvmaze", connec)



df['text'] = df['text'].apply(lemmatize_text)

import pickle
with open("lemmatize.pkl", "wb") as file:
    pickle.dump(df, file)