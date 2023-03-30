from base import *
import openai
import os, sys
import json 


with open("rsc/fun.dat", "r") as f:
    key = f.read().strip()
openai.api_key = key

with open("config.json", "r") as f:
    config = json.loads(f.read())


doc_files = os.listdir(config['out_path'] + 'docs/')
docs, doc_tags = load_docs(doc_files, config['out_path'] + 'docs/')


input_text = "Hello."

header = "You work for Unforgettable.me a private data-aggregation service that places the value of data in the hands of the user. Your job is to greet new participants and help them get started with the app.\nInput:\n"

context = "\n Notes:\n" + docs['welcome']

resp_text, ps, toks = probeGPT(header + input_text + context, "", model=model_map['ChatGPT'], temp=0.1)

print(resp_text)
