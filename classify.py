#!/usr/bin/env python
# coding: utf-8


import keras.models
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Classify script.')
parser.add_argument('--input-file',type=str,required=True, help='Path to the input text file.')
parser.add_argument('--encoding',type=str,default='utf-8', help='Text encoding of the inputfile. Default is utf-8.')
parser.add_argument('--output-json-file',type=str,required=True, help='Filename for the output JSON file.')
parser.add_argument('--explanation-output-dir',type=str,required=False, help='Directory for the explanation outputs.')
args = parser.parse_args()

model = keras.models.load_model("genre_final")

import pickle

with open("genre.pkl", "rb") as f:
    lookup = pickle.load(f)

print(lookup)

file = args.input_file
with open(file, 'r') as f:
    desp = f.read()



pred = model.predict([desp])
test = pd.DataFrame(data= pred,columns=lookup.keys())



print(test.apply(lambda row: row.nlargest(3), axis=1))

import json 
result = test.apply(lambda row: row.nlargest(3), axis=1)
result = result.columns.tolist()
with open(args.output_json_file, 'w') as file:
    json.dump(result, file)



