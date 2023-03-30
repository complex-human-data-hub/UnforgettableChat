import os, sys
import openai
import numpy as np


def load_docs(doc_files, doc_path):
    doc_tags = []
    docs = {}
    for i in range(len(doc_files)):
        doc_file = doc_files[i]
        with open(doc_path + doc_file, "r") as f:
            docs[doc_file.replace('.txt', '')] = f.read()
        doc_tags.append(doc_file.replace('.txt', ''))

    return docs, doc_tags


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if text != '':
        try:
            x = np.array(openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'])
        except Exception as e:
            print(e)
            x = np.zeros(1536)
    else:
        x = np.zeros(1536)
    return x
    
    
def probeGPT(cue=None, 
             system_text=None, 
             exemplar = {'question':'How are you?', 
                          'answer':"I'm well"}, 
             messages= None, 
             model='text-davinci-001', 
             temp=0.1):

    if model in ['gpt-3.5-turbo']:
        #chat models
        if messages is None:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"system", "content":system_text},
                          {"role": "user", "content": exemplar['question']},
                          {"role": "assistant", "content": exemplar['answer']},
                          {"role": "user", "content": cue}], temperature=temp) 
        else:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages, temperature=temp)
        resp_text = resp['choices'][0]['message']['content']
        ps = None
        toks = None

    else:
        resp = openai.Completion.create(
          model=model,
          prompt=cue,
          temperature=temp,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          logprobs=1,
          logit_bias={"50256":-5}
        )

        resp_text = resp['choices'][0]['text']
        #
        
        ps = np.exp(np.array(resp['choices'][0]['logprobs']['token_logprobs']))
        toks = np.array(resp['choices'][0]['logprobs']['tokens'])
    
    return resp_text, ps, toks
    
model_map = {'Davinci':'text-davinci-003',
             'Babbage':'text-babbage-001',
             'Curie':'text-curie-001',
             'Ada':'text-ada-001',
             'ChatGPT':'gpt-3.5-turbo'}
