# -*- coding: utf-8 -*-

import os
from collections import defaultdict



class data_prep():
    
    def __init__(self, dir_dict):
        self.source_dir      = dir_dict["source_dir"]
        self.reference_dir   = dir_dict["reference_dir"]
        self.system_dir      = dir_dict["system_dir"]
        self.eval_dir        = dir_dict["eval_dir"]
        self.output_dir      = dir_dict["output_dir"]
    
    
    def get_eval_files(self, dir_name):
        """
        Collect path of the files that contain evaluation scores.
        """
        files_dict  = {}
        
        for file in os.listdir(dir_name):
            if file.endswith(".csv"):                  
                if file.find("ad-seg-scores") != -1:                                      
                    tr = file[-9:-4]          #Name of the translation direction, e.g ru-en
                    files_dict[tr] = file
                                                                        
        return files_dict
        
    
    def get_source_ref_files(self, dir_name, avail_tr_list):
        """
        Collect path of the source files that have evaluation scores.
        """
        
        files_dict  = {}
        for file in os.listdir(dir_name):
            tr = file[13:17]
            tr = tr[0:2]+"-"+ tr[2:4]
            if tr in avail_tr_list:
                files_dict[tr] = file
        return files_dict    
    
    
    def get_system_ref_files(self, dir_name, avail_tr_list):
        """
        Collect path of the system/reference files that have evaluation scores.
        """
        
        files_dict  = defaultdict(lambda: defaultdict(str))
        
        for tr in avail_tr_list:
            for file in os.listdir(dir_name+tr):
                file_len   = len(file)
                model_name = file[13:(file_len-6)]
                files_dict[tr][model_name] = file
                #print("tr: {}, model_name: {}, file:{}".format(tr, model_name, file))
                                                    
        return files_dict
    
    def read_file(self, file_path, isFirstLine = False):
        """
        Read from a text file.
        """
        
        reader = open(file_path, "r", encoding="utf8")
        sent_dict = {}    
        i = 1 # Line starts at 1
        for line in reader:
            if not isFirstLine: # not reading the first line. becuase this line reads the column names.
                sent_dict[i] = line.strip()
                i += 1
            else:
                isFirstLine = False

        reader.close()                
        return sent_dict
    
    def read_eval_file(self, file_path, isFirstLine = True):
        """
        Read a file that contains human evaluation.
        """
        
        reader = open(file_path, "r", encoding="utf8")
        data_dict_zscore = defaultdict(lambda: defaultdict(float))  
        data_dict_rawscore = defaultdict(lambda: defaultdict(float)) 
        i = 0 # Line starts at 1
        for line in reader:
            if not isFirstLine: # not reading the first line. becuase this line reads the column names.
                line = line.strip()
                tokens = line.split()
                if len(tokens) == 5:
                    system_name     = tokens[0].strip()
                    line_num        = int(tokens[1].strip())
                    z_norm_score    = float(tokens[3].strip())
                    raw_score       = float(tokens[2].strip())
                    data_dict_zscore[system_name][line_num] = z_norm_score
                    data_dict_rawscore[system_name][line_num] = raw_score
                i += 1
            else:
                isFirstLine = False

        reader.close()                
        return data_dict_zscore, data_dict_rawscore
    
    def correct_system_name(self, systems_dict, isPrint=False):
        """
        Correct name of some systems that are named differently in the evaluation file.
        """
        
        for sys_name, sys_file in systems_dict.items():
            if sys_name == "rug_kken_morfessor.6677":
                del systems_dict[sys_name]
                sys_name_new = "rug-kken-morfessor.6677"  
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file 
            elif sys_name == "DBMS-KU_KKEN.6726":
                del systems_dict[sys_name]
                sys_name_new = "DBMS-KU-KKEN.6726" 
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file 
            elif sys_name == "talp_upc_2019_kken.6657":
                del systems_dict[sys_name]
                sys_name_new = "talp-upc-2019-kken.6657"
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file                 
            elif sys_name == "Frank_s_MT.6127":
                del systems_dict[sys_name]
                sys_name_new = "Frank-s-MT.6127" 
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file 
            elif sys_name == "Helsinki_NLP.6889":
                del systems_dict[sys_name]
                sys_name_new = "Helsinki-NLP.6889"    
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file 
            elif sys_name == "Helsinki_NLP.6889":
                del systems_dict[sys_name]
                sys_name_new = "Helsinki-NLP.6889"    
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file 
            elif sys_name == "Ju_Saarland.6525":
                del systems_dict[sys_name]
                sys_name_new = "Ju-Saarland.6525"     
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file 
            elif sys_name == "Facebook_FAIR.6937":
                del systems_dict[sys_name]
                sys_name_new = "Facebook-FAIR.6937"
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file 
            elif sys_name == "aylien_mt_gu-en_multilingual.6826":
                del systems_dict[sys_name]
                sys_name_new = "aylien-mt-gu-en-multilingual.6826"
                if isPrint: print("Corrected file from {} to {}".format(sys_name, sys_name_new))
                systems_dict[sys_name_new] = sys_file 
                        
        return systems_dict
        

    def create_files(self, files_dict, isPrint=True):
        """
        Create the mapping files. Each file contains source, reference, and system output as well as the evaluation scores.
        """
        
        eval_dict        = files_dict["eval_dict"]
        source_dict      = files_dict["source_dict"]
        reference_dict   = files_dict["reference_dict"]
        system_dict_all  = files_dict["system_dict"]
        eval_tr          = eval_dict.keys()  #Number of translations that have human evaluation
        
        for tr in eval_tr:
            # Create output dir
            path = self.output_dir + tr
            os.mkdir(path)
            
            
            # Get file paths
            src_file   = self.source_dir + source_dict[tr]
            ref_file   = self.reference_dir  +  reference_dict[tr]
            eval_file  = self.eval_dir  + eval_dict[tr]
            systems_dict = system_dict_all[tr]  # The system files under translation "tr"
            
            # Manual correction (Same system name is found named differently in the evaluation file)
            systems_dict = self.correct_system_name(systems_dict)
            
            # Read files
            src_sent_dict = self.read_file(src_file)
            ref_sent_dict = self.read_file(ref_file)
            eval_out_dict, eval_raw_dict = self.read_eval_file(eval_file) #gives human evaluation score for each system
            num_sent = len(ref_sent_dict.keys())
            
            #print("Tr: {}, length: {}, Sys in Eval: {}".format(tr, len(eval_out_dict.keys()), sorted(eval_out_dict.keys())))
            #print("Tr: {}, length: {}, Sys in Sys dir: {}".format(tr, len(systems_dict.keys()), sorted(systems_dict.keys())))
            for sys_name, sys_file in systems_dict.items():
                
                sys_sent_dict = self.read_file(self.system_dir + tr + "/" + sys_file)
                
                #cross checking that all three files have same length
                assert len(src_sent_dict.keys()) == len(ref_sent_dict.keys()) == len(sys_sent_dict.keys())
                
                
                if sys_name in eval_out_dict:
                    file_name = tr.replace("-","")+ "_" + sys_name + ".txt"
                    file_path = path + "/" +   file_name                  
                    file_obj = open(file_path, "w")
                    delim = "\t"
                    
                    
                    # Column line
                    line = "line_number" + delim + "source_sentence" + delim + "reference_sentence" + delim + "system_output" + delim + "z_score" + delim + "raw_score"
                    file_obj.write(line)
                    file_obj.write("\n")
                
                
                    # Get z-score and write into file
                    num_score_avail   = 0                                        
                    for line_num in range(1, num_sent+1): #line number starts at 1 and ends at num_sent                        
                        if line_num in eval_out_dict[sys_name]:
                            z_score   = eval_out_dict[sys_name][line_num]
                            raw_score = eval_raw_dict[sys_name][line_num]
                            num_score_avail += 1
                        else:
                            z_score    = "NA"
                            raw_score  = "NA"

                        line = str(line_num) + delim + src_sent_dict[line_num] + delim + ref_sent_dict[line_num] + delim + sys_sent_dict[line_num] + delim + str(z_score) + delim + str(raw_score)
                        file_obj.write(line)
                        file_obj.write("\n")
                    file_obj.close()
                    if isPrint: print("Translation direction: {}, System Name: {}, Num. Sentences: {}, Num. Sent. with score: {}, %Sent. with score: {}".format(tr, sys_name, num_sent, num_score_avail, num_score_avail*100.0/line_num ))
                
                else:
                    if isPrint: print("No Evaluation is found for- Translation {}, System: {}".format(tr, sys_name))
            if isPrint: print("\n")
                    
                
            
            
if __name__ == "__main__":
    
    # create directory:
    os.mkdir("./output/map_scores")
    os.mkdir("./output/map_scores/2019/")
    os.mkdir("./output/map_scores/2018/")
        
    for year in [2019, 2018]:
        print("\nStarted - create files for WMT-{}-------------------------------\n".format(year))
        
        if year == 2019:
            #2019_________________________
            wmt_dir          = "./data/wmt/wmt19-submitted-data/txt/"
            source_dir       = wmt_dir + "sources/"
            reference_dir    = wmt_dir + "references/"
            system_dir       = wmt_dir + "system-outputs/newstest2019/"
            eval_dir         = "./data/wmt/newstest2019-humaneval/mturk-sntlevel-humaneval-newstest2019/analysis/"            
            output_dir       = "./output/map_scores/2019/"    
        elif year == 2018:    
            #2018_________________________
            wmt_dir          = "./data/wmt/wmt18-submitted-data/txt/"
            source_dir       = wmt_dir + "sources/"
            reference_dir    = wmt_dir + "references/"
            system_dir       = wmt_dir + "system-outputs/newstest2018/"
            eval_dir         = "./data/wmt/newstest2018-humaneval/analysis/"
            output_dir       = "./output/map_scores/2018/"
        
        
        # Get the directory
        dir_dict  = {"source_dir":source_dir, "reference_dir":reference_dir, "system_dir":system_dir, "eval_dir":eval_dir, "output_dir":output_dir}    
        obj = data_prep(dir_dict)
        
        # Get the files that has human evaluation
        eval_dict = obj.get_eval_files(eval_dir)
        eval_tr = eval_dict.keys()  # Translation directions that have human evaluation
        
        # Get source files
        source_dict = obj.get_source_ref_files(source_dir, eval_tr)
        
        # Get reference files
        reference_dict = obj.get_source_ref_files(reference_dir, eval_tr)        
        
        # Get system files        
        system_dict = obj.get_system_ref_files(system_dir, eval_tr)        
        
        # Store all files into a dictionary
        files_dict = {"source_dict":source_dict, "reference_dict":reference_dict, "system_dict":system_dict, "eval_dict":eval_dict}
        
        # Create mapping files
        obj.create_files(files_dict)
        print("\nEnded - create files for WMT-{}-------------------------------\n\n".format(year))
    
    
    