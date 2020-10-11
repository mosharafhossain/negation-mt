# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import tensorflow as tf
import argparse
import json 
import cue_detection_mt

import pickle
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model

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
    
    # python extract_negation.py 
    argParser = argparse.ArgumentParser()    
    argParser.add_argument("-c", "--config_path", help="path to the configuration file of the Cue Detector", default="./negation_cue/config/config.json")
    argParser.add_argument("--out_dir", help="path to the output directory", default="./output/")

    args = argParser.parse_args()
    config_path = args.config_path
    out_dir     = args.out_dir

    
    # Read parameters from json file
    with open(config_path.strip()) as json_file_obj: 
        params = json.load(json_file_obj)

       
    # Set the seed    
    set_seed(params["seed"]) 


    # Load pre-trained model
    custom_objects={'CRF': CRF,'crf_loss':crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
    model = load_model(params["final_model_path"], custom_objects = custom_objects)

    
    # Load vocabs
    with open(params["vocab_loc"], "rb") as file_obj:
        vocab_dict = pickle.load(file_obj)  
    output_dict = {"model":model, "token_dict":vocab_dict["token_dict"], "index_dict":vocab_dict["index_dict"], "features_dict":vocab_dict["features_dict"], "max_len":vocab_dict["max_len"] }
    
    
    
    #Extract negation (more precisely, with and without negation) and token-level cue information files.
    mt_obj = cue_detection_mt.wmt_cue_detection()
    
    #2019
    file_output_dir = os.path.join(out_dir, "map_scores/2019")
    info_dir        = os.path.join(out_dir, "info")
    os.mkdir(info_dir)
    info_dir_wmt19  = os.path.join(info_dir, "2019")
    os.mkdir(info_dir_wmt19)
    mt_obj.create_all_neg_noneg_files(output_dict, file_output_dir, info_dir_wmt19)
        
    #2018
    file_output_dir = os.path.join(out_dir, "map_scores/2018")
    info_dir_wmt18  = os.path.join(info_dir, "2018")
    os.mkdir(info_dir_wmt18)
    mt_obj.create_all_neg_noneg_files(output_dict, file_output_dir, info_dir_wmt18)

    
