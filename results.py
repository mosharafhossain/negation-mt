# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:51:55 2020

@author: Mosharaf
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ranksums, mannwhitneyu
import scipy.stats as stats
import argparse

class data_structure():
    def __init__(self, tokens_list):
              
        self.pair_name         = tokens_list[0].strip()
        self.model_name        = tokens_list[1].strip()  
        # z-score
        self.z_score_all       = float(tokens_list[6])
        self.z_score_w_neg     = float(tokens_list[7])
        self.z_score_wo_neg    = float(tokens_list[8])
        # raw-score
        self.raw               = float(tokens_list[9])
        self.raw_wneg          = float(tokens_list[10])
        self.raw_woneg         = float(tokens_list[11])
        # SacreBLEU-score
        self.bleu              = float(tokens_list[12])
        self.bleu_wneg         = float(tokens_list[13])
        self.bleu_woneg        = float(tokens_list[14])
        # chrF++-score
        self.chrf              = float(tokens_list[15])
        self.chrf_wneg         = float(tokens_list[16])
        self.chrf_woneg        = float(tokens_list[17])
        # METEOR-score
        self.meteor            = float(tokens_list[18]) 
        self.meteor_wneg       = float(tokens_list[19])
        self.meteor_woneg      = float(tokens_list[20])
        

        
class data_prep():
    def data_read(self, file_name, isHeader=True):
        """
        Read a text file and create a list of objects. Each object is an instance of the data_structure class.
        params:
            @file_name: (string), name of the file
        return:
            obj_list: (List), a list of objects
        """
        
        reader = open(file_name, "r", encoding="utf8")
        pair_dict = {}
        obj_list = []
        token_list = []
        
        for line in reader: 
            if isHeader:
                isHeader = False
                continue
            token_list = line.split("\t")
            if len(token_list) >1:
                #print( "len: {}, tokens: {}".format(len(token_list), token_list  ))          
                obj = data_structure(token_list)
                obj_list.append(obj)
                token_list = []   
                pair_name = obj.pair_name
            else: 
                pair_dict[pair_name] = obj_list
                obj_list = []
                                
        reader.close()                
        return pair_dict


    def print_data(self, pair_dict):
        for key, obj_list in pair_dict.items():
            print("pair: {}, Number of Systems: {}________".format(key, len(obj_list)))
            for obj in obj_list:
                print("Tr: {}, Model Name: {}".format(key, obj.model_name))
                print("z: {}, z (w neg): {}, z (wo neg): {}".format(obj.z_score_all, obj.z_score_w_neg, obj.z_score_wo_neg))
                print("raw: {}, raw (w neg): {}, raw (wo neg): {}".format(obj.raw, obj.raw_wneg, obj.raw_woneg))
                print("bleu: {}, bleu (w neg): {}, bleu (wo neg): {}".format(obj.bleu, obj.bleu_wneg, obj.bleu_woneg))
                print("chrf: {}, chrf (w neg): {}, chrf (wo neg): {}".format(obj.chrf, obj.chrf_wneg, obj.chrf_woneg))
                print("meteor: {}, meteor (w neg): {}, meteor (wo neg): {}".format(obj.meteor, obj.meteor_wneg, obj.meteor_woneg))
                print("\n")
            print("\n\n")
            
            
    def evaluate(self, data_dict):
        """
        Evaluate the best submission (normalized direct assessments, Z) for each language direction 
        using all sentences, sentences with negation (w/ neg.), and sentences without negation (w/o neg.).
        """
        
        for pair in data_dict.keys():
            best_system   = data_dict[pair][0]
            #sys_name      = best_system.model_name
            z_all         = np.round(best_system.z_score_all, 3)
            z_w_neg       = np.round(best_system.z_score_w_neg, 3)
            z_wo_neg      = np.round(best_system.z_score_wo_neg, 3)
            delta1        = np.round( (z_w_neg-z_all)/z_all*100, 1)
            delta2        = np.round( (z_wo_neg-z_all)/z_all*100, 1)
            print("Lang pair: {}, z(all): {}, z(w/ neg): {}, delta: {}, z(w/o neg): {}, delta: {}".format(pair, z_all, z_w_neg, delta1, z_wo_neg, delta2 ))
            
    #Table 2_________________________________________________________________________
    def correlation(self, df, method_name="kendall"):
        """
        Compute correlations given a pandas dataframe using pandas libray. Available
        correlation types: kendall, pearson, and spearman
        """
        wneg  = df.loc[:,"z"].corr(df.loc[:,"z_wneg"], method=method_name)
        woneg = df.loc[:,"z"].corr(df.loc[:,"z_woneg"], method=method_name)
        corr = {"wneg":round(wneg,3), "woneg":round(woneg,3)}
        return corr
    
    
    def correlation_(self, df):
        """
        Compute correlations given a pandas dataframe using stats package. 
        """
        z_all = list(df.z)
        wneg  = list(df.z_wneg)
        woneg = list(df.z_woneg)
        tau_wneg, p_value_wneg = stats.kendalltau(z_all, wneg)
        tau_woneg, p_value_woneg = stats.kendalltau(z_all, woneg)
        tau_w_wo_neg, p_value_w_wo_neg = stats.kendalltau(wneg, woneg)
        
        corr = {"wneg":round(tau_wneg,3), "woneg":round(tau_woneg,3), "w_wo_neg":round(tau_w_wo_neg,3)}
        p_val = {"wneg":round(p_value_wneg,3), "woneg":round(p_value_woneg,3), "w_wo_neg":round(p_value_w_wo_neg,3)}
        corr_pval = {"corr":corr, "p_val":p_val}
        return corr_pval
    
    
    def get_kendall_corr(self, pair_dict):
        """
        generate kendall's correlation for all language directions.
        Args:
            @pair_dict: (Dictionary), keys is a language direction (e.g ru-en), 
                        value is list of objects of data_structure class.
                        
        """
        corr_dict = {}
        for key, obj_list in pair_dict.items():
            #print("\npair: {}, Number of Systems: {}".format(key, len(obj_list)))
            df = pd.DataFrame()
            z       = []
            z_wneg  = []
            z_woneg = []
            for obj in obj_list: # obj_list has values of all the systems who participated for a language direction.
                z.append(obj.z_score_all)
                z_wneg.append(obj.z_score_w_neg)
                z_woneg.append(obj.z_score_wo_neg)            
            #print("z: len: {}, val: {}".format(len(z), z))
            #print("z_wneg: len: {}, val: {}".format(len(z_wneg), z_wneg))
            #print("z_woneg: len: {}, val: {}".format(len(z_woneg), z_woneg))
            df["z"]       = z
            df["z_wneg"]  = z_wneg
            df["z_woneg"] = z_woneg
            #corr_dict[key] = self.correlation(df)
            corr_dict[key] = self.correlation_(df)
        return corr_dict
    
    
    def format_corr_results(self, corr_dict):
        
        print("\nCorrelation co-efficients------------------------------")
        for lang, val_dict in corr_dict.items():
            print("{}::   all v w/ neg: {},  all v w/o neg: {}, w/ neg v w/o neg: {}".format(lang, val_dict["corr"]["wneg"], val_dict["corr"]["woneg"], val_dict["corr"]["w_wo_neg"]))
            
        print("\nCorrelation p-values ------------------------------")
        for lang, val_dict in corr_dict.items():
            print("{}::   all v w/ neg: {},  all v w/o neg: {}, w/ neg v w/o neg:{}".format(lang, val_dict["p_val"]["wneg"], val_dict["p_val"]["woneg"], val_dict["p_val"]["w_wo_neg"]))
    
    # Table 3___________________________________________________________
    def get_rank(self, tuples):
        """
        Get the rank of each system.
        Arg:
            @tuples: (List), each element of the list is a tuple of (system_name, score) type.
        """
        rank = {}
        for i in range(len(tuples)):
            rank[tuples[i][0]] = i+1 #rank start with 1
        
        return rank
    
    def final_rank(self, all_list, rank_dict, rank_change_dict):
        """
        Generate final ranking. 
        
        Each output tuple is of type (rank, system_name, z_score, rank change).
        """
        result   = []
        result_s = [] #string output
        for t in all_list:
            sys_name = t[0]
            z_score  = t[1]
            new_t = (rank_dict[sys_name], sys_name, z_score, rank_change_dict[sys_name])
            new_t_s = (str(rank_dict[sys_name]) + " & " +str(rank_change_dict[sys_name]) + " & " +sys_name + " & " + str(z_score))
            result.append(new_t)
            result_s.append(new_t_s)
        return result, result_s
    
    def final_rank_v2(self, all_list, rank_dict, rank_change_dict):
        """
        Generate final ranking. 
        
        Each output tuple is of type (rank, system_name, z_score, rank change).
        """
        result   = []
        result_s = [] #string output
        for t in all_list:
            sys_name = t[0]
            z_score  = t[1]
            new_t = (rank_dict[sys_name], sys_name, z_score, rank_change_dict[sys_name])
            
            if rank_change_dict[sys_name] == 0:                
                rank_change = "--"
            elif rank_change_dict[sys_name] > 0:
                rank_change = "\\up{"+str(rank_change_dict[sys_name])+"}"
            elif rank_change_dict[sys_name] < 0:
                rank_change = "\\down{"+str(abs(rank_change_dict[sys_name]))+"}"
            new_t_s = (str(rank_dict[sys_name]) + " & " +str(rank_change) + " & " +sys_name + " & " + str(z_score))
            result.append(new_t)
            result_s.append(new_t_s)
        return result, result_s
    
    
    def ranking_systems(self, pair_dict):
        """
        Generate new ranking on z-score with and without negation.
        Args:
            @pair_dict: (Dictionary), keys is a language direction (e.g ru-en), 
                        value is list of objects of data_structure class.
        """
        rank_dict = {}
        rank_dict_s = {}
        final_dict_s = {}
        for key, obj_list in pair_dict.items():
            #print("\npair: {}, Number of Systems: {}".format(key, len(obj_list)))
            
            #Step 1: get the scores
            all_dict   = {}
            wneg_dict  = {}
            woneg_dict = {}            
            for obj in obj_list:
                all_dict[obj.model_name]   = obj.z_score_all
                wneg_dict[obj.model_name]  = obj.z_score_w_neg
                woneg_dict[obj.model_name] = obj.z_score_wo_neg
                
                
            #Step 2: get the ranking for the models based on avera z-score for all sentences, sentences with negation, and sentences without negation.
            all_list = sorted(all_dict.items(), key=lambda x:x[1], reverse=True) #List of tuples
            wneg_list = sorted(wneg_dict.items(), key=lambda x:x[1], reverse=True)
            woneg_list = sorted(woneg_dict.items(), key=lambda x:x[1], reverse=True)
            
            
            
            all_rank   = self.get_rank(all_list) #Dictionary, ranked for all systems
            wneg_rank  = self.get_rank(wneg_list)
            woneg_rank = self.get_rank(woneg_list)
                
            #Step 3: get rank change
            all_rank_change = {}
            wneg_rank_change = {}
            woneg_rank_change = {}
            for obj in obj_list: 
                all_rank_change[obj.model_name]   = all_rank[obj.model_name] - all_rank[obj.model_name]
                wneg_rank_change[obj.model_name]  = all_rank[obj.model_name] - wneg_rank[obj.model_name]
                woneg_rank_change[obj.model_name] = all_rank[obj.model_name] - woneg_rank[obj.model_name]
            
            all_, all_s = self.final_rank_v2(all_list, all_rank, all_rank_change)
            wneg_, wneg_s = self.final_rank_v2(wneg_list, wneg_rank, wneg_rank_change)
            woneg_, woneg_s = self.final_rank_v2(woneg_list, woneg_rank, woneg_rank_change)
            
            rank_dict[key] = {"all_":all_, "wneg_":wneg_, "woneg_":woneg_}
            rank_dict_s[key] = {"all_s":all_s, "wneg_s":wneg_s, "woneg_s":woneg_s}
            
            final_s = []
            for i in range(len(all_)):
                line = "& " + all_s[i].replace("& -- &", " & ") + " & " + wneg_s[i] + " & " + woneg_s[i] + "  \\"
                final_s.append(line)
            final_dict_s[key] = final_s
            
        return rank_dict, rank_dict_s, final_dict_s
    
    # Table 4__________________________________________________________________
    def get_comparison(self, pair_dict):
        """        
        Comparison between z-score and others (SacreBLEU, chrF, METEOR)
        Args:
            @pair_dict: (Dictionary), keys is a language direction (e.g ru-en), 
                        value is list of objects of data_structure class.
                        
        """
        corr_dict = {}
        for key, obj_list in pair_dict.items():
            #print("\npair: {}, Number of Systems: {}".format(key, len(obj_list)))
            df = pd.DataFrame()
            z          = []
            z_wneg     = []
            z_woneg    = []
            bleu       = []
            bleu_wneg  = []
            bleu_woneg = []
            chrf       = []
            chrf_wneg  = []
            chrf_woneg = []
            meteor       = []
            meteor_wneg  = []
            meteor_woneg = []
            
            for obj in obj_list: # obj_list has values of all the systems who participated for a language direction.
                #z
                z.append(obj.z_score_all)
                z_wneg.append(obj.z_score_w_neg)
                z_woneg.append(obj.z_score_wo_neg)    
                
                #bleu
                bleu.append(obj.bleu)
                bleu_wneg.append(obj.bleu_wneg)
                bleu_woneg.append(obj.bleu_woneg)
                
                #chrf
                chrf.append(obj.chrf)
                chrf_wneg.append(obj.chrf_wneg)
                chrf_woneg.append(obj.chrf_woneg)
                
                #meteor
                meteor.append(obj.meteor)
                meteor_wneg.append(obj.meteor_wneg)
                meteor_woneg.append(obj.meteor_woneg)
                
            #print("z: len: {}, val: {}".format(len(z), z))
            #print("z_wneg: len: {}, val: {}".format(len(z_wneg), z_wneg))
            #print("z_woneg: len: {}, val: {}".format(len(z_woneg), z_woneg))
            df["z"]            = z
            df["z_wneg"]       = z_wneg
            df["z_woneg"]      = z_woneg
            df["bleu"]         = bleu
            df["bleu_wneg"]    = bleu_wneg
            df["bleu_woneg"]   = bleu_woneg
            df["chrf"]         = chrf
            df["chrf_wneg"]    = chrf_wneg
            df["chrf_woneg"]   = chrf_woneg
            df["meteor"]       = meteor
            df["meteor_wneg"]  = meteor_wneg
            df["meteor_woneg"] = meteor_woneg
            
            # correlation________________
            # z with BLEU
            z_bleu_wneg     = df.loc[:,"z_wneg"].corr(df.loc[:,"bleu_wneg"], method="kendall")
            z_bleu_woneg    = df.loc[:,"z_woneg"].corr(df.loc[:,"bleu_woneg"], method="kendall")
            # z with chrF++
            z_chrf_wneg     = df.loc[:,"z_wneg"].corr(df.loc[:,"chrf_wneg"], method="kendall")
            z_chrf_woneg    = df.loc[:,"z_woneg"].corr(df.loc[:,"chrf_woneg"], method="kendall")
            # z with METEOR
            z_meteor_wneg   = df.loc[:,"z_wneg"].corr(df.loc[:,"meteor_wneg"], method="kendall")
            z_meteor_woneg   = df.loc[:,"z_woneg"].corr(df.loc[:,"meteor_woneg"], method="kendall")
            
            corr_dict[key] = {"z_bleu_wneg":round(z_bleu_wneg,3), "z_bleu_woneg":round(z_bleu_woneg,3),
                              "z_chrf_wneg":round(z_chrf_wneg,3), "z_chrf_woneg":round(z_chrf_woneg,3),
                              "z_meteor_wneg":round(z_meteor_wneg,3), "z_meteor_woneg":round(z_meteor_woneg,3)}
            
        return corr_dict
    
    
    def get_comparison_(self, pair_dict):
        """        
        Comparison between z-score and others (SacreBLEU, chrF, METEOR)
        Args:
            @pair_dict: (Dictionary), keys is a language direction (e.g ru-en), 
                        value is list of objects of data_structure class.
                        
        """
        corr_dict = {}
        pval_dict = {}
        val_dict  = {}
        for key, obj_list in pair_dict.items():
            #print("\npair: {}, Number of Systems: {}".format(key, len(obj_list)))
            #df = pd.DataFrame()
            z          = []
            z_wneg     = []
            z_woneg    = []
            bleu       = []
            bleu_wneg  = []
            bleu_woneg = []
            chrf       = []
            chrf_wneg  = []
            chrf_woneg = []
            meteor       = []
            meteor_wneg  = []
            meteor_woneg = []
            
            for obj in obj_list: # obj_list has values of all the systems who participated for a language direction.
                #z
                z.append(obj.z_score_all)
                z_wneg.append(obj.z_score_w_neg)
                z_woneg.append(obj.z_score_wo_neg)    
                
                #bleu
                bleu.append(obj.bleu)
                bleu_wneg.append(obj.bleu_wneg)
                bleu_woneg.append(obj.bleu_woneg)
                
                #chrf
                chrf.append(obj.chrf)
                chrf_wneg.append(obj.chrf_wneg)
                chrf_woneg.append(obj.chrf_woneg)
                
                #meteor
                meteor.append(obj.meteor)
                meteor_wneg.append(obj.meteor_wneg)
                meteor_woneg.append(obj.meteor_woneg)
                
            #print("z: len: {}, val: {}".format(len(z), z))
            #print("z_wneg: len: {}, val: {}".format(len(z_wneg), z_wneg))
            #print("z_woneg: len: {}, val: {}".format(len(z_woneg), z_woneg))
            
            # correlation________________
            # z with BLEU
            z_bleu_wneg, p_z_bleu_wneg       = stats.kendalltau(z_wneg, bleu_wneg)
            z_bleu_woneg, p_z_bleu_woneg     = stats.kendalltau(z_woneg, bleu_woneg)
            # z with chrF++
            z_chrf_wneg, p_z_chrf_wneg       =  stats.kendalltau(z_wneg, chrf_wneg)
            z_chrf_woneg, p_z_chrf_woneg     =  stats.kendalltau(z_woneg, chrf_woneg)
            # z with METEOR
            z_meteor_wneg, p_z_meteor_wneg   = stats.kendalltau(z_wneg, meteor_wneg)
            z_meteor_woneg, p_z_meteor_woneg = stats.kendalltau(z_woneg, meteor_woneg)
            
            corr_dict[key] = {"z_bleu_wneg":round(z_bleu_wneg,3), "z_bleu_woneg":round(z_bleu_woneg,3),
                              "z_chrf_wneg":round(z_chrf_wneg,3), "z_chrf_woneg":round(z_chrf_woneg,3),
                              "z_meteor_wneg":round(z_meteor_wneg,3), "z_meteor_woneg":round(z_meteor_woneg,3)}
            
            pval_dict[key] = {"z_bleu_wneg":round(p_z_bleu_wneg,3), "z_bleu_woneg":round(p_z_bleu_woneg,3),
                              "z_chrf_wneg":round(p_z_chrf_wneg,3), "z_chrf_woneg":round(p_z_chrf_woneg,3),
                              "z_meteor_wneg":round(p_z_meteor_wneg,3), "z_meteor_woneg":round(p_z_meteor_woneg,3)}
            
            val_dict[key]  = {"z_wneg":z_wneg, "z_woneg":z_woneg, 
                              "bleu_wneg":bleu_wneg,"bleu_woneg":bleu_woneg,
                              "chrf_wneg":chrf_wneg, "chrf_woneg":chrf_woneg,
                              "meteor_wneg":meteor_wneg, "meteor_woneg":meteor_woneg}
            
        corr_p_dict = {"corr_dict":corr_dict, "pval_dict":pval_dict, "val_dict":val_dict}
        
        return corr_p_dict
    
    def formatted_report(self, corr_dict_all):
        print("Lang Dir & bleu w/ neg. & bleu w/o neg. & chrF w/ neg. & chrF w/o neg. & meteor w/ neg. & meteor w/o neg. ")
        for key, corr_dict in corr_dict_all.items():            
            print("{} & {} & {} & {} & {} & {} & {}".format(key, corr_dict["z_bleu_wneg"], corr_dict["z_bleu_woneg"], corr_dict["z_chrf_wneg"], corr_dict["z_chrf_woneg"], corr_dict["z_meteor_wneg"], corr_dict["z_meteor_woneg"]))
    
    
    def wilcoxon_test(self, pair_dict):
        """
        The Wilcoxon signed-rank test is the non-parametric univariate test. 
        Data is not normally distributed.The paired observations are randomly and independently drawn.
        This test has no assumptions about the distribution of the data so no distribution check is needed.
        Consides median change rather than mean change.
        Do not require large sample sizes.
        Args:
            @pair_dict: (Dictionary), keys is a language direction (e.g ru-en), 
                        value is list of objects of data_structure class.
                        
        """
        corr_dict = {}
        for key, obj_list in pair_dict.items():
            print("\npair: {}, Number of Systems: {}".format(key, len(obj_list)))
            z_all   = []
            z_wneg  = []
            z_woneg = []
            for obj in obj_list: # obj_list has values of all the systems who participated for a language direction.
                z_all.append(obj.z_score_all)
                z_wneg.append(obj.z_score_w_neg)
                z_woneg.append(obj.z_score_wo_neg)            
            print("z_all: len: {}, val: {}".format(len(z_all), z_all))
            print("z_wneg: len: {}, val: {}".format(len(z_wneg), z_wneg))
            print("z_woneg: len: {}, val: {}".format(len(z_woneg), z_woneg))
            
            alternate = "greater" if np.median(z_all) > np.median(z_wneg) else "less"
            w_wneg, p_wneg  = wilcoxon(z_all, z_wneg, alternative=alternate) # median of z_all is greater than median of z_wneg
            
            alternate = "greater" if np.median(z_all) > np.median(z_woneg) else "less"
            w_woneg, p_woneg = wilcoxon(z_all, z_woneg, alternative=alternate)  # median of z_all is less than median of z_woneg
            
            
            alternate = "greater" if np.median(z_woneg) > np.median(z_wneg) else "less"
            w_wneg_woneg, p_wneg_woneg = wilcoxon(z_woneg, z_wneg, alternative=alternate)  # median of z_all is less than median of z_woneg
            
            
            corr_dict[key] = {"p_wneg":p_wneg, "p_woneg":p_woneg, "p_wneg_woneg":p_wneg_woneg}
            #print("tr: {}, z_all: {}".format(key, z_all))
            #print("tr: {}, z_wneg: {}".format(key, z_wneg))
            #print("tr: {}, z_woneg: {}\n".format(key, z_woneg))
            
        return corr_dict
    
    def ranksums_test(self, pair_dict):
        """
        generate kendall's correlation for all language directions.
        Args:
            @pair_dict: (Dictionary), keys is a language direction (e.g ru-en), 
                        value is list of objects of data_structure class.
                        
        """
        corr_dict = {}
        for key, obj_list in pair_dict.items():
            print("\npair: {}, Number of Systems: {}".format(key, len(obj_list)))
            z_all   = []
            z_wneg  = []
            z_woneg = []
            raw_all   = []
            raw_wneg  = []
            raw_woneg = []
            for obj in obj_list: # obj_list has values of all the systems who participated for a language direction.
                z_all.append(obj.z_score_all)
                z_wneg.append(obj.z_score_w_neg)
                z_woneg.append(obj.z_score_wo_neg)  
                
                raw_all.append(obj.raw)
                raw_wneg.append(obj.raw_wneg)
                raw_woneg.append(obj.raw_woneg)  
                
            print("z_all: len: {}, val: {}".format(len(z_all), z_all))
            print("z_wneg: len: {}, val: {}".format(len(z_wneg), z_wneg))
            print("z_woneg: len: {}, val: {}".format(len(z_woneg), z_woneg))
            print("raw_all: len: {}, val: {}".format(len(raw_all), raw_all))
            print("raw_wneg: len: {}, val: {}".format(len(raw_wneg), raw_wneg))
            print("raw_woneg: len: {}, val: {}".format(len(raw_woneg), raw_woneg))
            
            
            w_wneg, p_wneg  = ranksums(z_all, z_wneg)
            w_woneg, p_woneg = ranksums(z_all, z_woneg) 
            w_wneg_woneg, p_wneg_woneg = ranksums(z_wneg, z_woneg)  
            
            """
            w_wneg, p_wneg  = mannwhitneyu(z_all, z_wneg, alternative="less")
            w_woneg, p_woneg = mannwhitneyu(z_all, z_woneg, alternative="less") 
            w_wneg_woneg, p_wneg_woneg = mannwhitneyu(z_wneg, z_woneg, alternative="greater") 
            """
            #corr_dict[key] = {"all_wneg":round(p_wneg,3), "all_woneg":round(p_woneg,3), "wneg_woneg":round(p_wneg_woneg, 3)}
            corr_dict[key] = {"wneg_vs_woneg":round(p_wneg_woneg, 3)}
            
        return corr_dict
        
            
if __name__ == "__main__":
    
    #python results.py --table_no 1
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--result_dir", help="Path to the output directory", default="./output/results/")
    argParser.add_argument("--table_no", help="Any one of the reported tables (1 to 4)", default=1, type=int)
    args = argParser.parse_args()
    result_dir = args.result_dir
    table_no   = args.table_no
    prep_obj = data_prep()
    
    # WMT-2019
    file_name = os.path.join(result_dir, "model_ranking_wmt19.txt")
    pair_dict_19 = prep_obj.data_read(file_name)
    
    
    # WMT-2018
    file_name = file_name = os.path.join(result_dir, "model_ranking_wmt18.txt")
    pair_dict_18 = prep_obj.data_read(file_name)    
    
    if table_no == 1:
        print( "\n==================== WMT-2018 ==========================")
        prep_obj.evaluate(pair_dict_18)
        
        print( "\n==================== WMT-2019 ==========================")
        prep_obj.evaluate(pair_dict_19)
        
    elif table_no == 2:
        print( "\n==================== WMT-2018 ==========================")
        corr_dict_18 = prep_obj.get_kendall_corr(pair_dict_18)
        prep_obj.format_corr_results(corr_dict_18)
        
        
        print( "\n==================== WMT-2019 ==========================")
        corr_dict_19 = prep_obj.get_kendall_corr(pair_dict_19)
        prep_obj.format_corr_results(corr_dict_19)
    elif table_no == 3:
        print( "\n==================== WMT-2018 ==========================")
        rank_dict_18, rank_dict_18s, final_dict_18s = prep_obj.ranking_systems(pair_dict_18)        
        print(rank_dict_18["ru-en"])
        
        print( "\n==================== WMT-2019 ==========================")
        rank_dict_19, rank_dict_19s, final_dict_19s = prep_obj.ranking_systems(pair_dict_19)
        print(rank_dict_19["ru-en"])
    elif table_no == 4:
        print( "\n==================== WMT-2018 ==========================")
        corr_dict_18 = prep_obj.get_comparison_(pair_dict_18)
        tau_18       = corr_dict_18["corr_dict"]
        pvalue_18    = corr_dict_18["pval_dict"]
        print("coefficients: {}\n".format(tau_18))
        print("p-values: {}\n".format(pvalue_18))
        
        print( "\n==================== WMT-2019 ==========================")
        corr_dict_19 = prep_obj.get_comparison_(pair_dict_19)
        tau_19       = corr_dict_19["corr_dict"]
        pvalue_19    = corr_dict_19["pval_dict"]
        print("coefficients: {}\n".format(tau_19))
        print("p-values: {}\n".format(pvalue_19))
        
        

