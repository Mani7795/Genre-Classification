#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D, Input
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--training-data', type=str, required=True, help='\app')
args = parser.parse_args()

connec = sqlite3.connect(args.training_data)
df = pd.read_sql("select * from tvmaze", connec)

#df = pd.read_sql("Select t.description as text, GROUP_CONCAT(g.genre) as genres from tvmaze t join tvmaze_genre g on t.tvmaze_id = g.tvmaze_id where text is not NULL group by text", connec)



df3 = pd.read_sql("select * from tvmaze_genre", connec)

df = df.dropna().reset_index(drop=True)
mapping = df3.groupby('tvmaze_id')['genre'].apply(list).to_dict()
df['genre'] = df['tvmaze_id'].map(mapping)
import re
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def contains_link(text):
    if text is None:
        return False
    return bool(re.search(url_pattern, text))
df1 = pd.DataFrame()
# Apply the function to each row in the DataFrame
df1 = (df[~df['description'].apply(contains_link)])

#pd.read_sql("select genre from tvmaze_genre where tvmaze_id=1924", connec)



df1['description'] = df1['description'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
df1['description'] = df1['description'].str.lower()
df1
# from nltk.corpus import stopwords
# def remove_stopwords(words): 
#     """returns list of words without inconsequetial/unimportant/common words""" 
#     result= [] 
#     for w in words: 
#         if w not in stopwords.words("english"): 
#             result.append(w) 
#     return result 
# stop = (df['text'].apply(remove_stopwords))
# df1 = df[~stop]
# df1


def remove_short_words(text):
    words = text.split()  # Split the text into words
    words = [word for word in words if len(word) >= 3]  # Remove words with length less than 2
    return ' '.join(words)
df1['description'] = df1['description'].apply(remove_short_words)
df1['description']

custom_stopwords = ['a', 'about', 'above', 'after', 'again', 'all', 'am', 'an', 'are', 'aren', "aren't",
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'both', 'but', 'by', 'can', 'couldn', "couldn't",
    'd', 'did', 'didn', 'do', 'does', 'doesn', "doesn't", 'doing', "don't", 'down', 'during', 'each',
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


df1['description'] = df1['description'].apply(remove_custom_stopwords)


from collections import Counter
all_text = ' '.join(df1['description'])

# Split the text into words and count their occurrences
word_counts = Counter(all_text.split())

# Convert the word counts to a DataFrame
word_count_df = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Count'])

# Sort the DataFrame by word count in descending order
word_count_df = word_count_df.sort_values(by='Count', ascending=False)


final = word_count_df[word_count_df['Count']<=3]



single_words = set(final['Word'])

def single_freq(text):
    words = text.split()  # Split the text into words
    words = [word for word in words if word.lower() not in single_words]
    return ' '.join(words)


df1['description'] = df1['description'].apply(single_freq)

genre = df3['genre'].unique().tolist()
genre
df = df.dropna().reset_index(drop=True)
import numpy as np
target = np.zeros((df1.shape[0], len(genre)))
target.shape


categories_forward_lookup = {c:i for (i,c) in enumerate(genre)}
print(categories_forward_lookup)

for i, cs in zip(df.index, df.genre):
    for c in cs:
        category_number = categories_forward_lookup[c]
        target[i, category_number]= 1.0



from sklearn.model_selection import train_test_split
train_val_X, test_X, train_val_y, test_y = train_test_split(df1.description, target, test_size = 0.2)
train_X, validaton_X, train_y, validation_y = train_test_split(train_val_X, train_val_y, test_size=0.2)

max_tokens = 15000
output_sequence_length = 74
embedding_dim = 128


vectorizer = TextVectorization(max_tokens = max_tokens, output_sequence_length = output_sequence_length)
vectorizer.adapt(train_X)
inputs = Input(shape=(1,), dtype = tf.string)
vectorized = vectorizer(inputs)



model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_tokens,)),  # Input shape should match the number of features in TF-IDF
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(28, activation='softmax')  # Adjust num_classes accordingly
])


embedded = Embedding(max_tokens+1, embedding_dim)(vectorized)

averaged = GlobalAveragePooling1D()(embedded)


thinking = Dense(64, activation = 'relu')(averaged)
output = Dense(len(genre), activation = 'softmax')(thinking)



model = Model(inputs = [inputs], outputs = [output])


model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer= 'adam')


model.summary()


import keras.callbacks
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights = True)
history=model.fit(train_X, train_y,
                 validation_data=(validaton_X, validation_y), epochs=300, callbacks=callback)

model.evaluate(test_X,test_y)

test_df = pd.DataFrame(data=model.predict(test_X),columns=genre)
test_df

model.save("genre_final")



import pickle
with open("genre.pkl", "wb") as f:
    pickle.dump(categories_forward_lookup, f)




