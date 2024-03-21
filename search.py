import pickle
import argparse
parser = argparse.ArgumentParser(description='Searching Script.')
parser.add_argument('--input-file', type=str, required=True, help='Path to the input text file.')
parser.add_argument('--encoding', type=str, default= 'utf-8', help='Path to the encoding file.')
parser.add_argument('--output-json-file',type=str,required=True, help='Path to the output json file.')
args = parser.parse_args()

with open("lemmatize.pkl", "rb") as file:
    df = pickle.load(file)
import pandas as pd
from rank_bm25 import BM25Okapi
tokenized_docs = [doc.lower().split() for doc in df['text']]
bm25_obj = BM25Okapi(tokenized_docs)
def bm25_search(query, top_n=3):
    query_tokens = query.lower().split()
    scores = bm25_obj.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    top_results = df.loc[top_indices, ['tvmaze_id', 'showname']]
    return top_results

file_path = args.input_file
with open(file_path, "r") as file:
    content = file.read()
search_results = bm25_search(file_path, top_n=1)

print("Search Results:")
print(search_results)

import json

search_results_output = search_results.columns.tolist()
print(search_results_output)

with open(args.output_json_file, 'w') as json_file:
    json.dump(search_results_output, json_file, indent=4)