# -*- coding: utf-8 -*-
"""
"""
import negation_cue.cue_detector as cue_detector
import os
import random
import numpy as np
import tensorflow as tf
import argparse
import json 
import time
import pickle

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)	
    np.random.seed(seed)
    tf.set_random_seed(seed) # set the graph-level seed.

def read_file(file_path):
    sent_list = []    
    file_obj = open(file_path, "r", encoding="utf8")
    for line in file_obj:
        sent_list.append(line.strip())
    file_obj.close()                
    return sent_list

def write_file(file_path, sent_list):
    file_obj = open(file_path, "w")
    for sent in sent_list:
        file_obj.write(sent)
        file_obj.write("\n")
    file_obj.close()  

if __name__ == "__main__":
    
    # python train.py -c ./config/config.json 
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True)      
    args            = argParser.parse_args()
    config_path     = args.config_path
        
    
    # Read parameters from json file
    with open(config_path.strip()) as json_file_obj: 
        params = json.load(json_file_obj)

    # Set the seed    
    set_seed(params["seed"]) 
   
    # Train the model
    start = time.time()
    tr_model = cue_detector.train(params)
    isTrain = True
    output_dict = tr_model.fit(isTrain)
    stop = time.time()    
    #print("Model summary: {}".format(output_dict["model"].summary()))
    print("Training Time (in seconds): {}".format(stop - start))
    
    # Save vocab
    vocab_dict = {"token_dict":output_dict["token_dict"], "index_dict":output_dict["index_dict"], "features_dict":output_dict["features_dict"], "max_len":output_dict["max_len"]}
    with open(params["vocab_loc"], "wb") as file_obj:
        pickle.dump(vocab_dict, file_obj)
