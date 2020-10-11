# -*- coding: utf-8 -*-
"""
"""

import os
import negation_cue.cue_detector as cue_detector 
import shutil
import pickle

class data_structure():
    def __init__(self, tokens_list):
        self.line_number    = int(tokens_list[0].strip())   
        self.source_sent    = tokens_list[1].strip()
        self.reference_sent = tokens_list[2].strip()
        self.system_output  = tokens_list[3].strip()
        self.z_score        = tokens_list[4].strip()
        self.raw_score      = tokens_list[5].strip()

class wmt_cue_detection():
    
    def file_load(self, file_path, isColumnHeader = True):
        """
        Read a text file and produce a list of instances of data_structure class.
        """
        
        reader = open(file_path, "r", encoding="utf8")
        obj_list = []
        for line in reader: 
            if not isColumnHeader:
                tokens = line.split("\t")
                obj = data_structure(tokens)
                obj_list.append(obj)        
            else:
                isColumnHeader = False
        
        reader.close()                
        return obj_list
    
    def get_sentences(self, obj_list, sent_cat):
        sent_list = []
        for i in range(len(obj_list)):
            if sent_cat == "source":
                sent_list.append(obj_list[i].source_sent.replace("n’t", "n't"))
            elif sent_cat == "reference":
                sent_list.append(obj_list[i].reference_sent.replace("n’t", "n't"))
            elif sent_cat == "system":
                sent_list.append(obj_list[i].system_output.replace("n’t", "n't"))
            
        return sent_list
    
    def create_file(self, obj_list, indices, file_path):
        """
        Create files based on given indicies. A line number of a particular sentence is one more than the index of that sentence.
        """
        file_obj = open(file_path, "w")
        delim = "\t"
        z_score = []
        raw_score = []
        # Column line
        line = "line_number" + delim + "source_sentence" + delim + "reference_sentence" + delim + "system_output" + delim + "z_score" + delim + "raw_score"
        file_obj.write(line)
        file_obj.write("\n")
        
        for i in indices:
            line_num = i + 1 # Line number of a sentence is index + 1
            line = str(line_num) + delim + obj_list[i].source_sent + delim + obj_list[i].reference_sent + delim + obj_list[i].system_output + delim + str(obj_list[i].z_score) + delim + str(obj_list[i].raw_score)
            file_obj.write(line)
            file_obj.write("\n")
            if obj_list[i].z_score != "NA" and obj_list[i].raw_score != "NA":
                z_score.append(float(obj_list[i].z_score))
                raw_score.append(float(obj_list[i].raw_score))
        file_obj.close()
        
        return z_score, raw_score
    
    def save_in_pikle(self, info_dict, path):
        file_obj = open(path, "wb") 
        pickle.dump(info_dict, file_obj)        
        file_obj.close()
        print("Data created at {}".format(path))
        
                    
    def prepare_file(self, model_dict, file_name, file_dir, info_tr_dir):
        """
        Preapre neg and noneg files under each translation and system name
        params:
            @model_dict: trained model and related parameters.
            @file_name : Name of the original file for which neg and noneg files will be created.
            @file_dir  : Location of the translation directory where the original files are located
            @info_tr_dir: The translation directory where cue related information will be stored.
        Output:
            1. A new directory is created by the system/model name inside "file_dir" directory. Neg ang noneg files are created here. Also original file is moved here.
            2. cue_info_dict is saved in file and also return to the caller.
            cue_info_dict stores below informations:
            cue_info_dict: the values of the keys: "cue_info_src" or  "cue_info_ref" and "cue_info_sys" are also dictionary. Which contains below information
            cue_info_src["cues_count_dict"]: Cues count information. Keys: "singleword_cues", "prefix_cues", "suffix_cues", "multiwords_cues", "single_cues_sent", "multiple_cues_sent"
            cue_info_src["cues_dict"]: Cues and their counts. keys: "no", "n't", "not" etc..
            cue_info_src["indices"]: indices of sentences with/without neg cues. line number of a sentence can be get by add 1 to the index. keys: "neg_indices", "noneg_indices"
            cue_info_src["z_scores"]: z scores of human evaluation. keys:"neg", "noneg". Based on the paper z-score are normalised on the raw scores annotated by individual annotator?
            cue_info_src["raw_scores"]: raw_scores of human evaluation. keys:"neg", "noneg". 
        """        
        source_lang     = file_name[0:2]
        target_lang     = file_name[2:4]
        system_name     = file_name[0:-4]
        file_path       = file_dir + "/" + file_name
        obj_list        = self.file_load(file_path)

        
        #create output folder
        dir_path = file_path[0:-4]  #directory name is same as file name except the file extension
        os.mkdir(dir_path)
        pred_obj = cue_detector.cue_prediction()
        cue_info_dict = {}
        
        if source_lang == "en":
            src_sent_list = self.get_sentences(obj_list, "source")            
            cue_info_src = pred_obj.generate_cue_info(src_sent_list, model_dict)
            
            # file containing negation of source language
            neg_file = dir_path + "/" + system_name + "_src_neg.txt"
            z_score_neg, raw_score_neg = self.create_file(obj_list, cue_info_src["indices"]["neg_indices"], neg_file)            
            
            # file containing no negation of source language
            noneg_file = dir_path + "/" + system_name + "_src_noneg.txt"
            z_score_noneg, raw_score_noneg = self.create_file(obj_list, cue_info_src["indices"]["noneg_indices"], noneg_file)
            
            cue_info_src["z_scores"] = {"neg":z_score_neg, "noneg":z_score_noneg}
            cue_info_src["raw_scores"] = {"neg":raw_score_neg, "noneg":raw_score_noneg}
            cue_info_dict["cue_info_src"] = cue_info_src
            
        
        if target_lang == "en":
            
            # For Reference sentences___________________________________________
            ref_sent_list = self.get_sentences(obj_list, "reference")
            cue_info_ref = pred_obj.generate_cue_info(ref_sent_list, model_dict)
            
            # file containing negation of source language
            neg_file = dir_path + "/" + system_name + "_ref_neg.txt"
            z_score_neg, raw_score_neg = self.create_file(obj_list, cue_info_ref["indices"]["neg_indices"], neg_file)
            
            # file containing no negation of source language
            noneg_file = dir_path + "/" + system_name + "_ref_noneg.txt"
            z_score_noneg, raw_score_noneg = self.create_file(obj_list, cue_info_ref["indices"]["noneg_indices"], noneg_file)
            
            cue_info_ref["z_scores"] = {"neg":z_score_neg, "noneg":z_score_noneg}
            cue_info_ref["raw_scores"] = {"neg":raw_score_neg, "noneg":raw_score_noneg}
            cue_info_dict["cue_info_ref"] = cue_info_ref
        
        
            # For System output sentences___________________________________________
            sys_sent_list = self.get_sentences(obj_list, "system")
            cue_info_sys = pred_obj.generate_cue_info(sys_sent_list, model_dict)
            
            # file containing negation of source language
            neg_file = dir_path + "/" + system_name + "_sys_neg.txt"
            z_score_neg, raw_score_neg = self.create_file(obj_list, cue_info_sys["indices"]["neg_indices"], neg_file)

            
            # file containing no negation of source language
            noneg_file = dir_path + "/" + system_name + "_sys_noneg.txt"
            z_score_noneg, raw_score_noneg = self.create_file(obj_list, cue_info_sys["indices"]["noneg_indices"], noneg_file)

            cue_info_sys["z_scores"] = {"neg":z_score_neg, "noneg":z_score_noneg}
            cue_info_sys["raw_scores"] = {"neg":raw_score_neg, "noneg":raw_score_noneg}
            cue_info_dict["cue_info_sys"] = cue_info_sys
            
        
        #move the original file to the newly created system directory.         
        shutil.move(file_path, dir_path+"/"+file_name)   
        
        # Save cue_info_dict dictionary into pikle file  
        info_sys_dir = info_tr_dir + "/" + system_name 
        if not os.path.exists(info_sys_dir):
            os.mkdir(info_sys_dir)
        info_sys_path = info_sys_dir + "/" + system_name + ".pkl"
        self.save_in_pikle(cue_info_dict, info_sys_path)
        
        
        return cue_info_dict
            
    
    def create_all_neg_noneg_files(self, model_dict, file_output_dir, other_info_dir):
        """
        Create all files given a output_dir
        params:model_dict, contains the trained cue detector model and related parameters.
            @file_output_dir, the root directory where the original files with format(line, source, reference, system_output, z_score, raw_score) are located.
        """
        for tr in os.listdir(file_output_dir):
            tr_dir = file_output_dir + "/" + tr
            
            # create directory for saving dictionary information, dictionary contains all cue related information
            info_tr_dir = other_info_dir + "/" + tr
            if not os.path.exists(info_tr_dir):
                os.mkdir(info_tr_dir)
            
            print("Started for :{}".format(tr))
            for file_name in os.listdir(tr_dir):
                _ = self.prepare_file(model_dict, file_name.strip(), tr_dir, info_tr_dir)
                print("Completed for tr: {} and file: {}".format(tr, file_name))
                
        
        
        
        
        
        
        