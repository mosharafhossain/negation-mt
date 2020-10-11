# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 02:28:36 2020

@author: Mosharaf
"""

import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sacrebleu
import nltk.translate.meteor_score as ms
import chrF_pp # chrf++
from scipy.stats import norm
import argparse
from statsmodels.stats import weightstats as stests

np.random.seed(25)   


class data_structure():
    def __init__(self, tokens_list):
        self.line_number    = int(tokens_list[0].strip())   
        self.source_sent    = tokens_list[1].strip()
        self.reference_sent = tokens_list[2].strip()
        self.system_output  = tokens_list[3].strip()
        self.z_score        = tokens_list[4].strip()
        self.raw_score      = tokens_list[5].strip()


class data_preparation():
    
    def file_load(self, file_path, isColumnHeader = True):
        """
        Read lines from a text file. 
        Each line is tokenized and used to prepare a list of instances of the data_structure class.
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

        #get the source language in the file
        i = 0 if "/" not in file_path else (file_path.rindex("/")+1)
        tr_src = file_path[i:i+2]
                      
        return obj_list, tr_src
    
    def load_pickle_file(self, pkl_file):
        """
        Load a given pickle file into a dictionary. 
        """
        file_obj = open(pkl_file, "rb")
        cue_info_dict = pickle.load(file_obj)  #keys: tr_pred_scope,te_pred_scope
        file_obj.close()
        return cue_info_dict
    
    def get_length_bucket(self, sent_tokens):
        """
        Returns the appropriate length bucket for a given sentence.
        Params:
            @sent_tokens: (list), list of tokens of the given sentence.
        """
        sent_len = len(sent_tokens)
        if sent_len < 10:
            return "<10"
        elif sent_len >=10 and sent_len < 20:
            return "[10, 20)"
        elif sent_len >=20 and sent_len < 30:
            return "[20, 30)"
        elif sent_len >=30 and sent_len < 40:
            return "[30, 40)"
        elif sent_len >=40 and sent_len < 50:
            return "[40, 50)"
        elif sent_len >=50 and sent_len < 60:
            return "[50, 60)"
        elif sent_len >=60:
            return ">=60"
        
        
    
    def pred_info_sent(self, pred_sent, sent_tokens):
        """
        Produces information of a given sentence and a predicted outcome of that sentence.
        params:
            @pred_sent: (list), list of prediction of the tokens of a given sentence
            @sent_tokens: (list), list of tokens of a sentence.
        """
        
        sent_len = len(sent_tokens)  
        sent_bucket = self.get_length_bucket(sent_tokens)
        multi_words = ""
        neg_type = "N_C"
        neg_token = "NA"
        m_c = 0        
        for i in range(len(pred_sent)):                   
            if pred_sent[i] == "S_C":
                neg_type = "Single-word"
                neg_token = sent_tokens[i].lower()
                break
            elif pred_sent[i] == "PRE_C":
                neg_type = "Prefixal"
                neg_token = sent_tokens[i].lower()
                break
            elif pred_sent[i] == "POST_C":
                neg_type = "Suffixal"
                neg_token = sent_tokens[i].lower()
                break
            elif pred_sent[i] == "M_C":                    
                multi_words += sent_tokens[i].lower() + " "
                m_c += 1
                
                
        if m_c > 1:  
            neg_type = "Multi-word"
            neg_token = multi_words.strip()
            
        pred_info = {"sent_len":sent_len, "sent_bucket":sent_bucket, "neg_type":neg_type, "neg_token":neg_token}
        return pred_info
            
    def get_info(self, cue_info_dict, cue_info_key):
        """
        Generate information given a cue_info_dict.
        params:
            @cue_info_dict: (dictionary), contains necessary cue information of a particular system (who participated in WMT) for a language pair.
            @cue_info_key: (string), key that relates to a particular sentence category ("cue_info_src" or  "cue_info_ref" or "cue_info_sys")
            for example of the dictionary "cue_info_src":
                cue_info_src["cues_count_dict"]: Cues count information. Keys: "singleword_cues", "prefix_cues", "suffix_cues", "multiwords_cues", "single_cues_sent", "multiple_cues_sent"
                cue_info_src["cues_dict"]: Cues and their counts. keys: "no", "n't", "not" etc..
                cue_info_src["indices"]: indices of sentences with/without neg cues. line number of a sentence can be get by add 1 to the index. keys: "neg_indices", "noneg_indices"
                cue_info_src["z_scores"]: z scores of human evaluation. keys:"neg", "noneg". Based on the paper z-score are normalised on the raw scores annotated by individual annotator?
                cue_info_src["raw_scores"]: raw_scores of human evaluation. keys:"neg", "noneg". 
                cue_info_src["pred"]: (List of List), token-level prediction (S_C, PRE_C, POST_C, M_C) of each sentence
                cue_info_src["sent_tokens"]: (List of List), tokens of each sentence (tokenization is performed using spaCy)
        """
        line_number      = []
        neg_status       = []
        sent_len         = []
        sent_len_bucket  = []
        neg_type         = []
        neg_token        = []
        sent_tokens_list = []
        sent_size = len(cue_info_dict[cue_info_key]["sent_tokens"])
        for i in range(sent_size):
            line_number.append(i+1)
            
            pred_sent   = cue_info_dict[cue_info_key]["pred"][i]            
            sent_tokens = cue_info_dict[cue_info_key]["sent_tokens"][i]
            pred_info = self.pred_info_sent(pred_sent, sent_tokens)
            sent_len.append(pred_info["sent_len"])
            sent_len_bucket.append(pred_info["sent_bucket"])
            neg_type.append(pred_info["neg_type"])
            neg_token.append(pred_info["neg_token"])
            sent_tokens_list.append(sent_tokens)
            

            if i in cue_info_dict[cue_info_key]["indices"]["neg_indices"]:
                neg_status.append("neg")
            else:
                neg_status.append("noneg")
        
        info_dict =  {"line_number":line_number, "neg_status":neg_status, "sent_len":sent_len, "sent_len_bucket":sent_len_bucket, "neg_type":neg_type, "neg_token":neg_token, "sent_tokens":sent_tokens_list}
        return info_dict
    
    
    def calculate_bleu_sent(self, ref_sents, sys_sents):
        """
        Calculate BLEU by SacreBLEU sentence level by the guideline: https://github.com/mjpost/sacreBLEU.
        final score is 0 to 100.
        *Please note that sentence-level BLEU score is not a good way to report compared to document/corpus level BLEU.
        params:
            @ref_sents: (list), list of reference sentences/segments
            @sys_sents: (list), list of sentences by a system
        """
        bleu_scores = []
        for i in range(len(ref_sents)):
            reference  = [ref_sents[i]]
            system = [sys_sents[i]]
            bleu = sacrebleu.corpus_bleu(system, [reference])
            bleu_scores.append(round(bleu.score,3))
        return bleu_scores
    
    def calculate_chrf_sent(self, ref_sents, sys_sents):
        """
        Calculate sentence-level chrF++ by the guideline: https://github.com/m-popovic/chrF
        final score is 0 to 100.
        params:
            @ref_sents: (list), list of reference sentences/segments
            @sys_sents: (list), list of sentences by a system
        """
        scores = []
        for i in range(len(ref_sents)):
            reference  = [ref_sents[i]]
            system     = [sys_sents[i]]
            chrf = chrF_pp.corpus_chrf(reference, system)
            chrf = chrf["totalF"] #overall document/corpus level F-score (scaled to 100)
            scores.append(round(chrf,3))
        return scores
    
    def calculate_meteor_sent(self, ref_sents, sys_sents):
        """
        Calculate sentence level (not document or corpus level) METEOR score
        by the guideline: https://www.nltk.org/api/nltk.translate.html
        final score is 0 to 100.
        params:
            @ref_sents: (list), list of reference sentences/segments
            @sys_sents: (list), list of sentences by a system
        """
        scores = []
        for i in range(len(ref_sents)):
            reference  = [ref_sents[i]]
            system     = sys_sents[i]
            meteor     = ms.meteor_score(reference, system) * 100.0 # scale to 100
            scores.append(round(meteor,3))
            
        return scores
    
    
    def calculate_bleu(self, ref_sents, sys_sents):
        """
        Calculate SacreBLEU by using the guideline: https://github.com/mjpost/sacreBLEU.
        This is perfect way to calculate BLEU score.
        final score is 0 to 100.
        params:
            @ref_sents: (list), list of reference sentences/segments
            @sys_sents: (list), list of sentences by a system
        """
        bleu = sacrebleu.corpus_bleu(sys_sents, [ref_sents])
        return round(bleu.score,3)
    
    
    def calculate_chrf(self, ref_sents, sys_sents):
        """
        Calculate document level (aka corpus level) chrF++ (modified chrf) by using the guideline: https://github.com/m-popovic/chrF
        final score is 0 to 100.
        params:
            @ref_sents: (list), list of reference sentences/segments
            @sys_sents: (list), list of sentences by a system
        """
        chrf = chrF_pp.corpus_chrf(ref_sents, sys_sents)
        chrf = chrf["totalF"] #overall document/corpus level F-score (scaled to 100)
        return round(chrf,3)
    
        
    def prepare_info_dataframe(self, tr_src, cue_info_dict, obj_list):
        """
        Create a datafram using cue_info_dict and the score file for each system under a particular translation.
        params:
            @tr_src: (string), source language
            @cue_info_dict: (dictionary), contains cue info of a system under a particular translation.
            @obj_list: List of instances a score file contains
        """
        z_score     = [obj.z_score for obj in obj_list]
        raw_score   = [obj.raw_score for obj in obj_list]
        line_number = [obj.line_number for obj in obj_list]
        src_sent    = [obj.source_sent for obj in obj_list]
        ref_sent    = [obj.reference_sent for obj in obj_list]
        sys_sent    = [obj.system_output for obj in obj_list]
        bleu_score    = self.calculate_bleu_sent(ref_sent, sys_sent) # sentence level bleu score using SacreBLEU
        meteor_score  = self.calculate_meteor_sent(ref_sent, sys_sent) # sentence level meteor score using NLTK: ref https://www.nltk.org/api/nltk.translate.html
        #chrf_score    = self.calculate_chrf_sent(ref_sent, sys_sent) # sentence level chrf++ score using https://github.com/m-popovic/chrF
        
        if tr_src == "en":  #Source sentence in this translation is English
            
            # For Source Language
            info_dict_src = self.get_info(cue_info_dict, "cue_info_src")            
            assert line_number == info_dict_src["line_number"]
            
            
            
            df = pd.DataFrame({
                'line_number':line_number,
                'src_neg_status':info_dict_src["neg_status"],
                'src_sent_len':info_dict_src["sent_len"],
                'src_sent_bucket':info_dict_src["sent_len_bucket"],
                'src_neg_type':info_dict_src["neg_type"],
                'src_neg_token':info_dict_src["neg_token"],                
                'z_score':z_score,
                'bleu':bleu_score,
                'meteor':meteor_score,
                #'chrf':chrf_score,
                'raw_score':raw_score,
                'src_sent':src_sent,
                'ref_sent':ref_sent,
                'sys_sent':sys_sent                     
            })
                    

        else:
                        
            # Reference and System sentence is English
            # For Reference Translation
            info_dict_ref = self.get_info(cue_info_dict, "cue_info_ref")            
            
            # For System output
            info_dict_sys = self.get_info(cue_info_dict, "cue_info_sys") 
            
            #Validation
            assert line_number == info_dict_ref["line_number"]
            assert line_number == info_dict_sys["line_number"]
                    
            
            df = pd.DataFrame({
                'line_number':line_number,
                'ref_neg_status':info_dict_ref["neg_status"],
                'ref_sent_len':info_dict_ref["sent_len"],
                'ref_sent_bucket':info_dict_ref["sent_len_bucket"],
                'ref_neg_type':info_dict_ref["neg_type"],
                'ref_neg_token':info_dict_ref["neg_token"],
                'sys_neg_status':info_dict_sys["neg_status"],
                'sys_sent_len':info_dict_sys["sent_len"],
                'sys_sent_bucket':info_dict_sys["sent_len_bucket"],
                'sys_neg_type':info_dict_sys["neg_type"],
                'sys_neg_token':info_dict_sys["neg_token"],
                'z_score':z_score,
                'bleu':bleu_score,
                'meteor':meteor_score,
                #'chrf':chrf_score,
                'raw_score':raw_score,
                'src_sent':src_sent,
                'ref_sent':ref_sent,
                'sys_sent':sys_sent                                
            })
    
        return df
    
    def write_file(self, df, file_name):
        """
        Write the pandas dataframe into a text file.
        """
        columns = df.columns.values
        
        file_obj = open(file_name, "w", encoding="utf-8")
        delim = "\t"         
        column_header = ""
        for col_name in columns:
            column_header += col_name + delim       
        file_obj.write(column_header.strip())
        file_obj.write("\n")
        
        #print("column_header: {}".format(column_header))
        
        for i in range(df.shape[0]):
            row_str = ""
            for col in range(len(columns)):
                row_str += str(df.iloc[i, col]) + delim
            file_obj.write(row_str.strip())
            file_obj.write("\n")
        file_obj.close()
        
        

class reporting_tr():
    
    def p_values_ztest(self, v1, v2, sam_size=None):
        """
        Calculate p-value using z-test.
        Args:
            @var1: (List), all values for variable1
            @var2: (List), all values for variable2.
            @sam_size: (int), size of the sample.
        """
        mean_v1 = np.mean(v1)
        std_v1  = np.std(v1)
        
        mean_v2 = np.mean(v2)
        std_v2  = np.std(v2)
        
        if mean_v1 - mean_v2 > 0:
            mean_diff = mean_v1 - mean_v2
        else:
            mean_diff = mean_v2 - mean_v1
        
        std_diff = np.sqrt(std_v1**2 + std_v2**2)
        
        if sam_size:
            std_diff = std_diff/np.sqrt(sam_size)
        
        z_o = (0 - mean_diff)*1.0/std_diff
        p_value = norm.cdf(z_o)
        
        print ("Size- v1: {}, Mean- v1: {}, Std- v1: {}".format(len(v1), mean_v1, std_v1))
        print ("Size- v2: {}, Mean- v2: {}, Std- v2: {}".format(len(v2), mean_v2, std_v2))
        print ("mean_diff: {}, std_diff: {}, z_o: {}, p_value: {}\n".format(mean_diff, std_diff, z_o, p_value))
        
        ztest ,pval = stests.ztest(v1, v2, value=0)
        print("ztest: {}, pval: {}\n".format(ztest, pval))
        
        return  p_value
    
    def p_values_ttest(self, v1, v2):
        """
        Calculate the T-test for the means of two independent samples of scores. Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        Args:
            @var1: (List), all values for variable1
            @var2: (List), all values for variable2.
        """
        mean_v1 = np.mean(v1)
        std_v1  = np.std(v1)
        var_v1  = np.var(v1)
        
        mean_v2 = np.mean(v2)
        std_v2  = np.std(v2)
        var_v2  = np.var(v2)
        
        print ("Size- v1: {}, Mean- v1: {}, Std- v1: {}, Var- v1: {}".format(len(v1), mean_v1, std_v1, var_v1))
        print ("Size- v2: {}, Mean- v2: {}, Std- v2: {}, Var- v2: {}".format(len(v2), mean_v2, std_v2, var_v2))    
        diff1 = 0 if var_v2 == 0 else var_v1*1.0/var_v2
        diff2 = 0 if var_v1 == 0 else var_v2*1.0/var_v1
        print("var_v1/var_v2: {}, var_v2/var_v1: {}".format(diff1, diff2))
        result = stests.ttest_ind(v1, v2)
        ztest = result[0]
        p_value = result[1]
        print("t stat: {}, p_value: {}\n".format(ztest, p_value))
        
        
        return  p_value
    
    
    def get_p_values(self, var1, var2, perc_sample=0.80):
        """
        Calculate p-values for statistical significant test.
        Args:
            @var1: (List), all values for variable1
            @var2: (List), all values for variable2.
            @perc_sample: percentage of sample from the both variables
        Outputs:
            @p_value: (float), to test NULL hypothesis.
            @p_value_samp: (float), calculation considers the sample size. 
        """
        
        # Step 1: calculate p-values        
        p_value = self.p_values_ttest(var1, var2)
        #p_value = 0
        
        """
        # Step 2: calculate p-values (using sample size)
        l1 = len(var1)
        l2 = len(var2)
        if l1 > l2:
            size = round(perc_sample * l2)
        else:
            size = round(perc_sample * l1)
        
        ind_v1 = np.random.choice(l1, size, replace=False)
        ind_v2 = np.random.choice(l2, size, replace=False)
        
        p_value_samp = self.p_values(np.array(var1)[ind_v1], np.array(var2)[ind_v2], size)
        """
        p_value_samp = p_value
        
        return p_value, p_value_samp
        
        
    
    def get_scores(self, df, tr_sent_type):
        """
        Calculate z-score and BLEU
        params:
            @df: (DataFrame), contains information for all sentences for a particular system under a translation.
            @tr_sent_type: (string), neg/noneg segregation on which sentence category (source, reference, or system)
        """

        # Step 1: create dataframes_______________________________________________________
        df_eval = df[(df.z_score != "NA")]  # df for which the human evaluation exists      
        if tr_sent_type == "ref":            
            neg_df = df[(df.z_score != "NA") & (df.ref_neg_status == "neg")]
            noneg_df = df[(df.z_score != "NA") & (df.ref_neg_status == "noneg")]  
            neg_df_all = df[(df.ref_neg_status == "neg")]
            noneg_df_all = df[(df.ref_neg_status == "noneg")]    
        elif tr_sent_type == "sys":
            neg_df = df[(df.z_score != "NA") & (df.sys_neg_status == "neg")]
            noneg_df = df[(df.z_score != "NA") & (df.sys_neg_status == "noneg")]
            neg_df_all = df[(df.sys_neg_status == "neg")]
            noneg_df_all = df[(df.sys_neg_status == "noneg")]
        elif tr_sent_type == "src":   
            neg_df = df[(df.z_score != "NA") & (df.src_neg_status == "neg")]
            noneg_df = df[(df.z_score != "NA") & (df.src_neg_status == "noneg")]
            neg_df_all = df[(df.src_neg_status == "neg")]
            noneg_df_all = df[(df.src_neg_status == "noneg")]
            
            
        # Step 2. Count___________________________________________________________________
        sent_count = df.shape[0]
        parc_sent_eval  = round( (df_eval.shape[0] * 100.0)/sent_count, 2) #Parcentag of sentences that have human evaluation.
        parc_sent_wneg  = round( (neg_df.shape[0] * 100.0)/df_eval.shape[0], 2) #Parcentag of neg sentences that have human evaluation.
        parc_sent_woneg = round( (noneg_df.shape[0] * 100.0)/df_eval.shape[0], 2) #Parcentag of neg sentences that have human evaluation.
    
        # Step 3. Z-Score_________________________________________________________________
        z_score = round(np.mean(  [ float(z) for z in df_eval.z_score] ), 3)  #all sentences who have human evaluation
        z_wneg  = round(np.mean(  [ float(z) for z in neg_df.z_score] ), 3)   #sentences with negation who have human evaluation
        z_woneg = round(np.mean(  [ float(z) for z in noneg_df.z_score] ), 3) #sentences without negation who have human evaluation
        
        # Step 4. Raw human evaluation score_________________________________________________________________
        raw_score = round(np.mean(  [ float(z) for z in df_eval.raw_score] ), 3)  #all sentences who have human evaluation
        raw_wneg  = round(np.mean(  [ float(z) for z in neg_df.raw_score] ), 3)   #sentences with negation who have human evaluation
        raw_woneg = round(np.mean(  [ float(z) for z in noneg_df.raw_score] ), 3) #sentences without negation who have human evaluation
        
        
        
        
        # Step 5. SacreBLEU score (for the sentences where human evaluation is available)______
        prep_obj = data_preparation()        
        ref_sents = [ s for s in df_eval.ref_sent]
        sys_sents = [ s for s in df_eval.sys_sent]
        bleu = prep_obj.calculate_bleu(ref_sents, sys_sents)
        
        ref_sents = [ s for s in neg_df.ref_sent]
        sys_sents = [ s for s in neg_df.sys_sent]
        bleu_wneg = prep_obj.calculate_bleu(ref_sents, sys_sents)
        
        ref_sents = [ s for s in noneg_df.ref_sent]
        sys_sents = [ s for s in noneg_df.sys_sent]
        bleu_woneg = prep_obj.calculate_bleu(ref_sents, sys_sents)
        
        
        # Step 6. SacreBLEU score (for all the sentences at the same time)______________________________________        
        #This is just for test purpose.
        ref_sents = [ s for s in df.ref_sent]
        sys_sents = [ s for s in df.sys_sent]
        bleu_all = prep_obj.calculate_bleu(ref_sents, sys_sents)
        
        ref_sents = [ s for s in neg_df_all.ref_sent]
        sys_sents = [ s for s in neg_df_all.sys_sent]
        bleu_wneg_all = prep_obj.calculate_bleu(ref_sents, sys_sents)
        
        ref_sents = [ s for s in noneg_df_all.ref_sent]
        sys_sents = [ s for s in noneg_df_all.sys_sent]
        bleu_woneg_all = prep_obj.calculate_bleu(ref_sents, sys_sents)
        
        # Step 7. SacreBLEU score (first sentence level then take the average). asl means average over sentence level. 
        #This is just for test purpose.
        bleu_all_asl = round(np.mean([ b for b in df.bleu]),3)
        bleu_wneg_all_asl = round(np.mean([ b for b in neg_df_all.bleu]),3)
        bleu_woneg_all_asl = round(np.mean([ b for b in noneg_df_all.bleu]),3)
        
        
        #Step 8. chrF score (for the sentences where human evaluation is available)______
        ref_sents = [ s for s in df_eval.ref_sent]
        sys_sents = [ s for s in df_eval.sys_sent]
        chrf = prep_obj.calculate_chrf(ref_sents, sys_sents)
        
        ref_sents = [ s for s in neg_df.ref_sent]
        sys_sents = [ s for s in neg_df.sys_sent]
        chrf_wneg = prep_obj.calculate_chrf(ref_sents, sys_sents)
        
        ref_sents = [ s for s in noneg_df.ref_sent]
        sys_sents = [ s for s in noneg_df.sys_sent]
        chrf_woneg = prep_obj.calculate_chrf(ref_sents, sys_sents)
        
        
        
        #Step 9. meteor score (for the sentences where human evaluation is available)______
        meteor       = round(np.mean([ b for b in df_eval.meteor]),3)
        meteor_wneg  = round(np.mean([ b for b in neg_df.meteor]),3)
        meteor_woneg = round(np.mean([ b for b in noneg_df.meteor]),3)

               
        scores = {"sent_count":sent_count, "parc_sent_eval":parc_sent_eval, "parc_sent_wneg":parc_sent_wneg, "parc_sent_woneg":parc_sent_woneg, 
                  "z_score":z_score, "z_wneg":z_wneg, "z_woneg":z_woneg, 
                  "raw_score":raw_score, "raw_wneg":raw_wneg, "raw_woneg":raw_woneg, 
                  "bleu":bleu, "bleu_wneg":bleu_wneg, "bleu_woneg":bleu_woneg, 
                  "bleu_all":bleu_all, "bleu_wneg_all":bleu_wneg_all, "bleu_woneg_all":bleu_woneg_all, 
                  "bleu_all_asl":bleu_all_asl, "bleu_wneg_all_asl":bleu_wneg_all_asl, "bleu_woneg_all_asl":bleu_woneg_all_asl, 
                  "chrf":chrf, "chrf_wneg":chrf_wneg, "chrf_woneg":chrf_woneg,
                  "meteor":meteor, "meteor_wneg":meteor_wneg, "meteor_woneg":meteor_woneg}
        
        
        return scores
    
    
    def tr_wise_analysis(self, score_dir, info_dir, isPrint=True):
        """
        This function produces translation direction wise and system wise score information.
        params:
            @score_dir: (string), the durectory where score files resides
            @info_dir: (string), the durectory where information files resides
        """
        prep_obj = data_preparation()        
        scores_dict = defaultdict(lambda: defaultdict(dict) )
        
        for tr in os.listdir(info_dir):
            info_tr_dir   = info_dir + "/" + tr
            score_tr_dir  = score_dir + "/" + tr            
            
            for dir_name in os.listdir(info_tr_dir):
                info_path     = info_tr_dir + "/" + dir_name + "/" + dir_name + ".pkl"
                score_path    = score_tr_dir + "/" + dir_name + "/" + dir_name + ".txt"                
                
                # Step 1: Load score file into list of data structure object. tr_src is the source language in the translation
                obj_list, tr_src = prep_obj.file_load(score_path)
    
                # Step 2: Load pickle information file that contains cue information
                cue_info_dict = prep_obj.load_pickle_file(info_path)    
                
                # Step 3: Create Dataframe using obj_list and cue_info
                df =  prep_obj.prepare_info_dataframe(tr_src, cue_info_dict, obj_list)
                
                
                # Step 5: Reporting                
                if tr_src == "en": # if Source language is English                                        
                    tr_sent_type = "src"    
                    scores_src = self.get_scores(df, tr_sent_type)   
                    scores = {"src":scores_src}
                else: # If Source is not English
                    
                    # Reference Translation__________________________________                    
                    tr_sent_type = "ref"
                    scores_ref = self.get_scores(df, tr_sent_type)                    
                    
                    
                    # System output
                    tr_sent_type = "sys"
                    scores_sys = self.get_scores(df, tr_sent_type)  
                    
                    scores = {"ref":scores_ref, "sys":scores_sys}
                    
                scores_dict[tr][dir_name] = scores
                
                if isPrint: print("\nCompleted for: Tr: {}, System Name: {}\n".format(tr, dir_name))
                
        return scores_dict
    
    def write_into_file(self, scores_dict, file_path, isSentTypeSys=False):
        """
        Write the scores in a text file.
        params:
            @scores_dict: (Dictionary), contains the scores.
            @file_path: (string), path of the output file.
            @isSentTypeSys: (Boolean), this is set to True if target language is English and negation split is on System output.
        """
        file_obj = open(file_path, "w")
        delim = "\t"
        
                
        column = "trans. name" + delim + "model name" + delim + "num. of sent." + delim \
                 + "% sent with eval." + delim + "%sent with neg" + delim + "%sent without neg" + delim \
                 + "Z-score" + delim + "Z-score(w_neg)" + delim + "Z-score(wo_neg)" + delim \
                 + "raw-score" + delim + "raw-score(w_neg)" + delim + "raw-score(wo_neg)" \
                 + delim + "BLEU" + delim + "BLEU(w_neg)" + delim + "BLEU(wo_neg)" + delim \
                 + "chrF" + delim + "chrF(w_neg)" + delim + "chrF(wo_neg)"+ delim \
                 + "METEOR" + delim + "METEOR(w_neg)" + delim + "METEOR(wo_neg)"
        file_obj.write(column)
        file_obj.write("\n")
        
        model_scores = {}
        for tr, score_dict in scores_dict.items():
            # Step 0. Select the appropriate sentence type. if source is English then sent_type is "src". 
            # If English is a target language then sent_type is "ref" ("in this case sent_type can be "sys", however we want to split with neg/without neg in reference sentence). 
            if tr.strip()[0:2] == "en":
                sent_type = "src"
            else:
                if isSentTypeSys:
                    sent_type = "sys"
                else:
                    sent_type = "ref" 
            
            #Step 1: Get models and z-score to rank the models
            model_scores = {model:score_sent_dict[sent_type]['z_score'] for model, score_sent_dict in score_dict.items() }
            sorted_list = sorted(model_scores.items(), key=lambda x:x[1], reverse=True) # This results sorted tuples of model and Z-score
            
            for t in sorted_list:
                model = t[0]
                
                line = tr + delim + model[5:] + delim + str(score_dict[model][sent_type]["sent_count"]) + delim \
                                      + str(score_dict[model][sent_type]["parc_sent_eval"]) + delim \
                                      + str(score_dict[model][sent_type]["parc_sent_wneg"]) + delim \
                                      + str(score_dict[model][sent_type]["parc_sent_woneg"]) + delim \
                                      + str(score_dict[model][sent_type]["z_score"]) + delim \
                                      + str(score_dict[model][sent_type]["z_wneg"]) + delim \
                                      + str(score_dict[model][sent_type]["z_woneg"]) + delim \
                                      + str(score_dict[model][sent_type]["raw_score"]) + delim \
                                      + str(score_dict[model][sent_type]["raw_wneg"]) + delim \
                                      + str(score_dict[model][sent_type]["raw_woneg"]) + delim \
                                      + str(score_dict[model][sent_type]["bleu"]) + delim \
                                      + str(score_dict[model][sent_type]["bleu_wneg"]) + delim \
                                      + str(score_dict[model][sent_type]["bleu_woneg"]) + delim \
                                      + str(score_dict[model][sent_type]["chrf"]) + delim \
                                      + str(score_dict[model][sent_type]["chrf_wneg"]) + delim \
                                      + str(score_dict[model][sent_type]["chrf_woneg"]) + delim \
                                      + str(score_dict[model][sent_type]["meteor"]) + delim \
                                      + str(score_dict[model][sent_type]["meteor_wneg"]) + delim \
                                      + str(score_dict[model][sent_type]["meteor_woneg"]) + delim 
                                      
                file_obj.write(line)
                file_obj.write("\n")
            
            file_obj.write("\n")
            
        file_obj.close()

if __name__ == "__main__":
    #python analysis.py 
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--out_dir", help="path to the output directory", default="./output/")
    args = argParser.parse_args()
    out_dir = args.out_dir
    
    analysis_dir = os.path.join(out_dir, "results")
    os.mkdir(analysis_dir)
    
    
    rep_tr_obj = reporting_tr()
    
    # For english to other language ditection: splits with and without negation samples on source language 
    # and for other to english direction: splits with and without negation samples  on reference translation.
    for year in [2019, 2018]:
        print("\nStarted - preparing ranking file for WMT-{}-------------------------------\n".format(year))
        if year == 2019:
            score_dir = os.path.join(out_dir, "map_scores/2019")
            info_dir  = os.path.join(out_dir, "info/2019")
                        
            scores_dict = rep_tr_obj.tr_wise_analysis(score_dir, info_dir)            
            file_path =  os.path.join(analysis_dir, "model_ranking_wmt19.txt") 
            rep_tr_obj.write_into_file(scores_dict, file_path)                    
        elif year == 2018:
            # For 2018
            score_dir = os.path.join(out_dir, "map_scores/2018")
            info_dir  = os.path.join(out_dir, "info/2018")
            
            scores_dict = rep_tr_obj.tr_wise_analysis(score_dir, info_dir)            
            file_path =  os.path.join(analysis_dir, "model_ranking_wmt18.txt")
            rep_tr_obj.write_into_file(scores_dict, file_path)
            
        print("\nEnded - preparing ranking file for WMT-{}-------------------------------\n".format(year))
                        

        
