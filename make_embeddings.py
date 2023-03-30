from base import *
import openai
import sys, os
import json


with open("rsc/fun.dat", "r") as f:
    key = f.read().strip()
openai.api_key = key

with open("config.json", "r") as f:
    config = json.loads(f.read())



docs, doc_tags = load_docs(config["out_path"])
