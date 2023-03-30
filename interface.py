from flask import Flask, render_template, request, jsonify
import json
from base import *
import openai
import os, sys
import json
import numpy as np



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global chat_exemplars, n_hist, history, M, thresh, intentKnown
    message = request.form['message']
    x = get_embedding(message)

    resonance = x.dot(M)
    if np.max(resonance) > thresh:
        recollection = history[np.argmax(resonance)]
    else:
        recollection = ""

    if False:
    #if (not intentKnown):
        intent, ps, toks = probeGPT(messages=intent_check_exemplars + [{'role':'user', 'content':message}], model=model_map['ChatGPT'], temp=0)
        print(intent)
        if intent in ['intro', 'configuration']:
            intentKnown = True
            instructions = docs[intent]
            header = '''You work for Unforgettable.me a private data-aggregation service that places the value of data in the hands of the user. Your job is to greet new participants and help them get started with the app. Use your notes to respond to the input step-by-step. You are a Chatbot that helps visitors to the Unforgettable.me website get started. ''' + instructions + '''General: Ask the user to reply after each step to give them the next step and tell them to ask you if they need more help'''

            chat_exemplars = [{'role':'system', 'content':header}]
            response = "Great! I know how to help you with that. " + docs[intent]
        else:
            response = "I'm sorry, I don't know how to help you with that. Please provide me more information about what you need help with."

    #if (not intentKnown):
    #    resp_text, ps, toks = probeGPT(messages=intent_exemplars, model=model_map['ChatGPT'], temp=0.7)    
    #else:
    if True:
        print(len(chat_exemplars))
        #print("Recollection: ", recollection )
        chat_exemplars.append({'role':'user', 'content': message})
        # Process the message and generate a response
        resp_text, ps, toks = probeGPT(messages=chat_exemplars, model=model_map['ChatGPT'], temp=0.1)
        chat_exemplars.append({'role':'assistant', 'content': resp_text})
        #mem = "User: " + message + "\nBot: " + resp_text + "\n"
        #history.append(mem) 
        #M[:, n_hist] = get_embedding(mem)

        response = resp_text#f"You said: {message}"
    return jsonify({'response': response})


with open("rsc/fun.dat", "r") as f:
    key = f.read().strip()
openai.api_key = key

with open("config.json", "r") as f:
    config = json.loads(f.read())


doc_files = os.listdir(config['out_path'] + 'docs/')
docs, doc_tags = load_docs(doc_files, config['out_path'] + 'docs/')

n_hist = 1
history = ['']
M = np.zeros((1536, 10000)) # conversation memory

thresh = 0.5
input_text = "Hello."

doc_tag = sys.argv[1]

instructions = docs[doc_tag]

header_intent = "You work for Unforgettable.me a private data-aggregation service that places the value of data in the hands of the user. Your job is to greet new participants and find out what they need help with. If they haven't registered yet, downloaded the app, and logged in, then they need the introductory guides. If they have, then they need help with configuration."
header_intent_check = "Your job is to help the program know the user's intent. If the user needs help getting registered and download the app, then their intent is 'intro'. If the participant needs to configure the settings on the app of phone, then their intent is 'configuration'. Your job is to see their message and provide a single word response: either 'intro', 'configuration', or 'unknown' if you do not know the user's intent. Respond with a single word."

intent_exemplars = [{'role':'system', 'content': header_intent}]
intent_check_exemplars = [{'role':'system', 'content': header_intent_check}]

header = '''Your name is Chester. You work for Unforgettable.me a private data-aggregation service that places the value of data in the hands of the user. Your job is to greet new participants and help them get started with the app. Use your notes to respond to the input step-by-step. You are a Chatbot that helps visitors to the Unforgettable.me website get started. ''' + instructions + '''General: Ask the user to reply after each step to give them the next step and tell them to ask you if they need more help'''

chat_exemplars = [{"role": "system", "content": header}] #+ chat_exemplars
intentKnown = False

if __name__ == '__main__':
    app.run(debug=True)

