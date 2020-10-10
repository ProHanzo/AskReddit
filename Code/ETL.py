############################ Import all Libraries #############################
import numpy as np 
import pandas as pd
import praw
from github import Github
from github import InputGitTreeElement
from datetime import datetime
import base64

# Tensorflow 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical


################################# Parameters #################################

# Location of the original dataset
original_dataset = "#####"

# Account information needed to log into Reddit Api to collect new Askreddit data
reddit_parameters = {'client_id':"#####",
                     'client_secret':"#####",
                     'username':"#####",
                     'password':"#####",
                     'user_agent':'#####',
                     'time':'week'}

# Classification model's parameters
classification_parameters = {'embedding_dim':16,
                             'learning_rate':0.01,
                             'Dropout':0.5,
                             'epochs':30}

# Generation model's parameters
generation_parameters = {'above_this_Upvote':3000,
                         'learning_rate':0.01,
                         'Dropout':0.3,
                         'epochs':60}

# Github login and other information to load the new dataset and ML models
github_parameters = {'user':'#####',
                     'password':'#####',
                     'repo_name':'#####',
                     'branch_name':'#####',
                     'commit_msg':'#####'}


################################### Extract ###################################

# Get original askreddit dataset from github
df = pd.read_csv(original_dataset)

# Use Reddit Api to collect data from /r/AskReddit
def collect_subreddit_data(acc_info, subreddit='askreddit', time='week'):
    '''
    Parameters
    ----------
    acc_info : Dict
        The dictionary account information to log into reddit api
    subreddit : String, optional
        Decide which subreddit to extract data from. The default is 'askreddit'.
    time : TYPE, optional
        Decide which timeframe to extract data from. The default is 'week'.

    Returns
    -------
    A pandas DataFrame containing the title and score data from the subreddit.

    '''
    try:
        reddit = praw.Reddit(client_id=acc_info['client_id'],
                         client_secret=acc_info['client_secret'],
                         password=acc_info['password'],
                         user_agent=acc_info['user_agent'],
                         username=acc_info['username'])
    except:
        print("Unable to connect to praw")
        pass
    
    # Collect Askreddit data using the subreddit and time parameter
    title = []
    score = []
    for submission in reddit.subreddit("askreddit").top(time,limit=100):
        title.append(submission.title)
        score.append(submission.score)
    
    # Add the titles and scores to a pandas DataFrame
    df_add = pd.DataFrame({'Title':title, 'Upvotes':score})
    
    # Add a Upvotes_Bin before returning the DataFrame
    bands = [0, 500, 1000, 5000, 10000, 50000, 1500000]
    new_labels = [0,1,2,3,4,5]
    
    df_add['Upvotes_Bin'] = df_add["Upvotes"]
    df_add['Upvotes_Bin'] = pd.cut(df_add["Upvotes_Bin"], bins=bands, labels=new_labels)
    
    return df_add

# Create a DataFrame of the new extracted data using the Reddit API
print("~~~Collecting Reddit Data from API~~~")
try:
    df_new = collect_subreddit_data(reddit_parameters, time=reddit_parameters["time"])
    print("Successfully collected new subreddit data")
except:
    print("Failed to collect new subreddit data")


################################## Transform ##################################

# Combine the new DataFrame to the original, then remove duplicates
df = pd.concat([df,df_new])
df.drop_duplicates(subset ="Title",keep = "first", inplace = True)
df.reset_index(drop=True, inplace=True)

# Train a new Classifier Model using the new Dataset
def train_classifier(df, params):
    '''
    Parameters
    ----------
    df : pandas DataFrame
        This DataFrame contains two columns, 'Title', and 'Upvotes'.
    params : Dict
        The parameters for the classification model.

    Returns
    -------
    The trained classification model.

    '''
    
    # Parameters #
    
    # The size of the numpy array based on the sentence
    embedding_dim = params['embedding_dim']
    # Max length of how long a sentence can be
    max_length = 13
    # Start from the end of the sentence and leave the remainder a 0
    trunc_type='post'
    # Start from the end of the sentence and leave the remainder a 0
    padding_type='post'
    # Replace Unknown words in a test set into this string
    oov_tok = "<OOV>"

    # Initiate the Tokenizer
    tokenizer = Tokenizer(oov_token=oov_tok)
    
    # Create a collection of words using the tokenizer
    corpus = df["Title"]
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    # pad the sentences from the DataFrame's Title into numpy arrays 
    sequences = tokenizer.texts_to_sequences(corpus)
    padded = pad_sequences(sequences, maxlen=max_length,padding=padding_type, truncating=trunc_type)
    
    np_padded = np.array(padded)
    np_labels = to_categorical(df["Upvotes_Bin"])
    
    # Train the Model
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(params['Dropout']),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
    ])
    
    adam = Adam(lr=params['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    model.fit(np_padded, np_labels, epochs=params['epochs'], verbose=1)
    
    return model
    
# Initiate the classification model
print("~~~Training Classification Model~~~")
try:
    classification_model = train_classifier(df, classification_parameters)
    print("Successfully trained classification model")
except:
    print("Failed to train classification model")
    
# Train a new Generation Model using the new Dataset    
def train_generation(df, params):
    '''
    Parameters
    ----------
    df : pandas DataFrame
        This DataFrame contains two columns, 'Title', and 'Upvotes'.
    params : Dict
        The parameters for the generation model.
        
    Returns
    -------
    The trained generation model.

    '''
    # Create a copy to use for the model and drop rows that have less than 3000 Upvotes
    df_g = df.copy()  
    df_g.drop(df_g.loc[df_g['Upvotes'] < params['above_this_Upvote']].index, inplace=True)
    
    # Initiate the Tokenizer
    tokenizer = Tokenizer()
    
    # Create a collection of words using the tokenizer
    corpus = df_g["Title"]   
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    # iterates through a sentence creating new labels which are the array's last index 
    for line in corpus:
    	token_list = tokenizer.texts_to_sequences([line])[0]
    	for i in range(1, len(token_list)):
    		n_gram_sequence = token_list[:i+1]
    		input_sequences.append(n_gram_sequence)

    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # create predictors and label
    xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
    
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    
    # Train the Model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dropout(params['Dropout']))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(lr=params['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    model.fit(xs, ys, epochs=params['epochs'], verbose=1)
    
    return model
    
# Initiate the generation model
print("~~~Training Generation Model~~~")
try:
    generation_model = train_generation(df, generation_parameters)
    print("Successfully trained Generation Model")
except:
    print("Failed to train Generation Model")

# Save the models
print("~~~Saving the new ML Models~~~")
try:
    classification_model.save("classification")
    generation_model.save("generation")
    print("Model Saving Successful")
except:
    print("Failed to Save Models")

################################## Load #######################################
    
# Log into github account
try:
    g = Github(github_parameters['user'],github_parameters['password'])
    print("Successfully log in to github")
except:
    print("Failed to log in to github")


# Prepare the new dataset
#convert pd.df to text. This avoids writing the file as csv to local and again reading it
df_file = df.to_csv(sep=',', index=False)

dataset_file_name = ["Data/askreddit_data_api.csv"]
dataset_repo_path = [df_file]
# Prepare the Classification Model
classification_file_name = ['Classification_Model/saved_model.pb','Classification_Model/variables/variables.data-00000-of-00001','Classification_Model/variables/variables.index']
classification_repo_path = ['classification/saved_model.pb','classification/variables/variables.data-00000-of-00001','classification/variables/variables.index']
# Prepare the Generation Model
generation_file_name = ['Generation_Model/saved_model.pb','Generation_Model/variables/variables.data-00000-of-00001','Generation_Model/variables/variables.index']
generation_repo_path = ['generation/saved_model.pb','generation/variables/variables.data-00000-of-00001','generation/variables/variables.index']
# Combine the Model's files and names
model_file_name = classification_file_name + generation_file_name
model_repo_path = classification_repo_path + generation_repo_path


# Function to update the new dataset to github
def update_dataset_files(file_names,file_list,git_account,Repo,branch,commit_message =""):
    # if commit message is empty enter the date instead
    if commit_message == "":
       commit_message = "Data Updated - "+ datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # log into the github account
    g = git_account
    # retrieve the github repo
    repo = g.get_user().get_repo(Repo)
    # connect to the repo's branch
    master_ref = repo.get_git_ref("heads/"+branch)
    master_sha = master_ref.object.sha
    base_tree = repo.get_git_tree(master_sha)
    # create a list and append the dataset files to it
    element_list = list()
    for i in range(0,len(file_list)):
        element = InputGitTreeElement(file_names[i], '100644', 'blob', file_list[i])
        element_list.append(element)
    # initiate a git tree and commit the new files to it
    tree = repo.create_git_tree(element_list, base_tree)
    parent = repo.get_git_commit(master_sha)
    commit = repo.create_git_commit(commit_message, tree, [parent])
    master_ref.edit(commit.sha)


def update_model_files(file_names,file_list,git_account,Repo,branch,commit_message=""):
    # if commit message is empty enter the date instead
    if commit_message == "":
        commit_message = "Data Updated - "+ datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # log into the github account
    g = git_account
    # retrieve the github repo
    repo = g.get_user().get_repo(Repo)
    # connect to the repo's branch
    master_ref = repo.get_git_ref("heads/"+branch)
    master_sha = master_ref.object.sha
    base_tree = repo.get_git_tree(master_sha)
    # create a list and append the model files to it
    element_list = list()
    for i in range(len(file_list)):
        data = base64.b64encode(open(file_list[i], "rb").read())
    
        blob = repo.create_git_blob(data.decode("utf-8"), "base64")
    
        element = InputGitTreeElement(path=file_names[i], mode='100644', type='blob', sha=blob.sha)
        element_list.append(element)
   
    # initiate a git tree and commit the new files to it
    tree = repo.create_git_tree(element_list, base_tree)
    parent = repo.get_git_commit(master_sha)
    commit = repo.create_git_commit(commit_message, tree, [parent])
    master_ref.edit(commit.sha)

# Update the new dataset to github
print("~~~Uploading new files to github~~~")
try:
    update_dataset_files(dataset_file_name,dataset_repo_path,g,github_parameters['repo_name'],github_parameters['branch_name'])
    print("Successfully updated the new dataset to github")
except:
    print("Failed to update the new dataset to github")
    
# Update the new model to github
try:
    update_model_files(model_file_name,model_repo_path,g,github_parameters['repo_name'],github_parameters['branch_name'])
    print("Successfully updated the new model to github")
except:
    print("Failed to update the new model to github")