Intially, before index we need to clean the dataset. Checking for null values, removing punctuations, symbols, nummbers as they do not have any effect on searching. Here lemmetization is used for indexing. Lemmetization is used to get the standardied words from their root words. This can help in improving our search analysis. The sentence passed from the description column is tokenized. Each word is tokenized and the sentence is split Wordnet Lemmmatizer is used for performing lemmatization. WordNet lemmatizer is used for lemmatization. wordnet.verb is used to get the part of speech of the word as verb.


Once the indexing is done using lemmatization, bm250kapi is used for searching and bm25 can also be used to do indexing based on the query which is provided. The tokenized text is passed into bm250kapi. This api is advanced version of tf-idf algorithm where it ranks the documents based on frequency of the words, using IDF it calculate the importance of the terms. 


```python

```
