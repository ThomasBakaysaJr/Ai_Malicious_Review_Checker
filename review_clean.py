# going to clean the reviews here.
import sys
import string
import json
import random
import time
import joblib
import gc
import os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.sparse import hstack
from io import StringIO

# clean and then predict. returns predictions
def clean_predict(in_chunk_df, vizer, bootstrap_model, threshold):
    in_chunk_df = in_chunk_df
    # clean text and then normalize rating
    in_chunk_df['text'] = in_chunk_df['text'].apply(clean_text)
    in_chunk_df['stars'] = in_chunk_df['stars'] / 5.0

    # vectorize chunk's text
    x_text = vizer.transform(in_chunk_df['text'])
    # convert starts (rating) to 2d array
    rate_feature = in_chunk_df['stars'].values.reshape(-1,1)
    
    # crate the hstack to be used in the support model
    X = hstack([x_text, rate_feature])
    
    # return predictions based on 'confidence' scores
    # 0 means the model is confident (above threshold) that it's a real review
    # 1 means 
    conf_scores = bootstrap_model.predict_proba(X)
    real_mask = np.where((conf_scores[:, 0] > threshold), 0, 2)
    fake_mask = np.where((conf_scores[:, 1] > threshold), 1, 2)

    y_pred = pd.DataFrame(np.where(real_mask == 0, 0, np.where(fake_mask == 1, 1, 2)))
    # rename column
    print(f'Values in this chunk.\n{y_pred.value_counts()}')
    
    return y_pred

# converts to lowercase and strip punctuation
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def main():
    # do get rid of annoying warnings
    pd.set_option('future.no_silent_downcasting', True)
    # global, for the print statements
    verbose = True
    # set to True so that the notebook tries smaller chunks and only does 5 chunks
    test = False
    # How confident does the model need to be to accept the psuedo-label
    threshold = 0.8
    # chunk_size (will only use this if test if False)
    CHUNK_SIZE = 100000
    vizer_path = 'models/support_vectorizer.pkl'
    bootstrap_model_path = 'models/support_log.pkl'
    
    # load the tf-idf vectorizer and support_svm
    try:
        vizer = joblib.load(vizer_path)
        bootstrap_model = joblib.load(bootstrap_model_path)
        print('Bootstrap model and vectorizer loaded.')
    except Exception as e:
        print(f"Can't load bootstrap model and vectorizer due to: {e}")
        print(f'Cannot continue without those.')
        sys.exit()
        
    # this should probably be on a seperate cell so I don't constantly reload the dataframes
    fDefPath = 'reviews/yelpReviews/yelp_academic_dataset_'
    # constants so I don't have to keep changing names
    BS = 'business'
    CH = 'checkin'
    TI = 'tip'
    RW = 'review'
    US = 'user'
    
    # subsets of what i care about
    bssub = ['business_id', 'postal_code',
             'review_count', 'attributes', 'categories']
    ussub = ['user_id', 'review_count', 'yelping_since']
    # this top one is for when we use for final training
    rwsub = ['user_id', 'business_id', 'stars', 'text', 'date']
    # for bootstrap model, we will need to link the user_ids to this for
    # when training the bigger model. For actual bootstrap training we'll
    # stip the user out of it
    rwsub_less = ['user_id','business_id','stars', 'text']
    
    # constants for the file path
    bspath = f'{fDefPath}{BS}.json'
    chpath = f'{fDefPath}{CH}.json'
    tipath = f'{fDefPath}{TI}.json'
    rwpath = f'{fDefPath}{RW}.json'
    uspath = f'{fDefPath}{US}.json'
    
    chunk_save_path = 'files/review_chunks/review_chunk_0'
    
    # going to have read the json file in chunks, the thing is almost 5 gigs
    chunk_size = 200 if test else CHUNK_SIZE
    
    with open(rwpath, 'r', encoding='utf-8') as file:
        chunk = []
        chunk_df = pd.DataFrame()
        chunk_count = []
        count = 1
        print(f'Starting labeling')
        t0 = time.time()
        # check if this chunk exists already (for restarts)
        chunk_path = f'{chunk_save_path}{count}.csv'
        
        if os.path.exists(chunk_path):
            chunk_count.append(count)
        
        for index, line in enumerate(file):            
            if count not in chunk_count:
                # read each line as a dataframe then append to a list
                data = json.loads(line)
                #data = pd.read_json(StringIO(line), lines = True)
                chunk.append(data)
            
            # check if chunk is full / right now we exit since I'm just trying to clean the thing rn.
            if (index + 1) % chunk_size == 0:
                if os.path.exists(chunk_path):
                    print(f'Chunk {count} already exits at {chunk_path}. Skipping')
                else:
                    print(f'Starting labeling of chunk {count}.')
                    chunk_df = pd.DataFrame(chunk)
                    
                    # remove the columns we don't care about
                    chunk_df = chunk_df[rwsub_less]
        
                    # predict pseudo labels
                    chunk_df = pd.concat([chunk_df, clean_predict(chunk_df,vizer, bootstrap_model, threshold)], axis = 1)
                    chunk_df.rename({0:'pseudo_label'}, inplace = True, axis='columns')
        
                    # remove entries that the model was not confident on (true or fake)
                    chunk_df = chunk_df[chunk_df['pseudo_label'] != 2]
                    
                    # write each chunk to its own file, will combine them later
                    chunk_df.to_csv(chunk_path, index=False)
        
                    if verbose:
                        print(f'chunk {count} finished at {time.time() - t0} seconds. Saved at {chunk_path}\n')
                        
                # garbage collection
                del chunk, chunk_df
                gc.collect()
        
                chunk = []
                chunk_df = pd.DataFrame()
                
                if count not in chunk_count:
                    chunk_count.append(count)
                count += 1
                chunk_path = f'{chunk_save_path}{count}.csv'
                
                if os.path.exists(chunk_path):
                    chunk_count.append(count)
                
                if test and count > 5:
                    break
    if test:
        print('TEST RUN')
    print(f'Finished labeling reviews after {(time.time() - t0) / 60.0} minutes. Files are seperated into chunks of {chunk_size} lines.')
    
if __name__ == "__main__":
    main()
