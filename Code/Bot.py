# Import all Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from subprocess import check_output
import os
import pandas as pd
import numpy as np
import praw

import stat
import shutil
from subprocess import call
from pathlib import Path

# Parameters
# Account information needed to log into Reddit Api to collect new Askreddit data
reddit_parameters = {'client_id':"#####",
                     'client_secret':"#####",
                     'username':"#####",
                     'password':"#####",
                     'user_agent':'#####',
                     'time':'week'}

# Function to delete the git repo folder
dir = str(os.getcwd()) + '/repo'
repo_path = Path(dir)

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)

def delete_dir(dir):
    for i in os.listdir(dir):
        if i.endswith('git'):
            tmp = os.path.join(dir, i)
            # We want to unhide the .git folder before unlinking it.
            while True:
                call(['attrib', '-H', tmp])
                break
            shutil.rmtree(tmp, onerror=on_rm_error)
            
    shutil.rmtree('repo')

# If git repo exists delete first before downloading the new repo
try:
    if(repo_path.exists()):
        delete_dir(dir)
        print("Successfully deleted git repo")
    else:
        print("Repo doesn't exist, Moving on")
except:
    print("Failed to delete git repo")

# Download the Git Repo where the ML Models are
print("~~~Downloading git repo~~~")
try:
    path = ' ' + str(os.getcwd()) + '/repo'
    cmd = 'git clone https://github.com/ProHanzo/AskReddit.git' + path
    check_output(cmd, shell=True).decode()
    print("Successfully downloaded the git repo")
except:
    print("Failed to download the git repo")


# Extract Classification Model
print("~~~Extracting Classification Model~~~")
try:
    classification_model = tf.keras.models.load_model('repo/Classification_Model')
    print("Successfully extracted classification model")
except:
    print("Failed to extract classification Model")
    

# Extract Generation Model
print("~~~Extracting Generation Model~~~")
try:
    generation_model = tf.keras.models.load_model('repo/Generation_Model')
    print("Successfully extracted generation model")
except:
    print("Failed to extract generation Model")


# Prepare the Models for prediction
# Extract the Askreddit dataset
link = "https://raw.githubusercontent.com/ProHanzo/AskReddit/master/Data/askreddit_data_api.csv"
df = pd.read_csv(link)

# initiate the Tensorflow's Tokenizer
tokenizer = Tokenizer()
# Set the initial parameters and dataset for the predictions
max_length = 13
corpus = df["Title"]
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
# Function 
input_sequences = []
for line in corpus:
    # Create a array of numbers from the sentences from the dataset's Title Column
	token_list = tokenizer.texts_to_sequences([line])[0]
    # Create a Numpy array from the sentence that leaves the last word from 
    # sentence as the label
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])

###### Generate n number of questions ######
# List of first word to generate the questions from
seed_list = ["what",'if','whats','you','how', 'who', 'why', 'when', 'where']
generation_list = []
next_words = 13

# Iterate through the seed_list to generate sentences
for i in seed_list:
	seed_text = ''
	seed_text += i
	for _ in range(next_words):
        # Continue to generate words using the current word as the feature
        # And the next word as the prediction
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = generation_model.predict_classes(token_list, verbose=0)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	generation_list.append(seed_text)

# Classify the questions and pick the best one
token_list = tokenizer.texts_to_sequences([generation_list[0]])
token_pad = pad_sequences(token_list, maxlen=next_words, padding='post')
    
prev = classification_model.predict(token_pad)
highest = prev
highest_indx = 0

# Iterate through the list of generated questions and pick the best one
for i in range(len(generation_list)):   

    # Prepare the questions to be classified
    token_list = tokenizer.texts_to_sequences([generation_list[i]])
    token_pad = pad_sequences(token_list, maxlen=max_length, padding='post')
    
    # Make the prediction
    prev = classification_model.predict(token_pad)
    
    # Highest number from the array
    prev_max = np.argmax(prev)
    highest_max = np.argmax(highest)
    
    # Pick the highest indexed question and from that index pick the highest score
    if((prev_max >= highest_max and prev[0,prev_max] >= highest[0,highest_max])):
        highest = prev
        highest_indx = i

    
# Pick the best question using the highest_indx
best_question = generation_list[highest_indx]
# Clean up the question to a proper form
best_question = (best_question + '?').capitalize()

# Log in to Reddit Api
try:
    reddit = praw.Reddit(client_id=reddit_parameters['client_id'],
                     client_secret=reddit_parameters['client_secret'],
                     password=reddit_parameters['password'],
                     user_agent=reddit_parameters['user_agent'],
                     username=reddit_parameters['username'])
except:
    print("Unable to connect to praw")
    
# Post the question on AskReddit
try:
    reddit.subreddit("Askreddit").submit(best_question, selftext='')
    print("Successfully posted the question on Askreddit")
except:
    print("Failed to post the question on Askreddit")

# Delete the git repo to clean disk space
try:
    if(repo_path.exists()):
        delete_dir(dir)
        print("Successfully deleted git repo")
    else:
        print("Repo doesn't exist, Moving on")
except:
    print("Failed to delete git repo")



