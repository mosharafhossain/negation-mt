# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import matplotlib.pyplot as plt


CUES = ["I_C"]
CUES_SPEC = ["S_C", "M_C", "PRE_C", "POST_C"] 
SCOPES = ["I_S"]
EVENTS = ["I_E"]
PREFIX_CUES = ['dis', 'im', 'in', 'ir', 'un' ]
SUFFIX_CUES = ['less', 'lessly', 'lessness']

global_pre_miss = 0
global_post_miss = 0

class evaluation():
    def predict_test(self, model, test_x, index2label):
        test_pred = model.predict(test_x)   # test_predict 3D array. 1st dim: number of rows, 2nd dim: label type index, 3rd dim: one-hot-vector of that label type
        test_pred = np.argmax(test_pred, axis=-1)  # applying argmax on last dimension. 
        test_pred = [[index2label[i] for i in sent] for sent in test_pred ]
        return test_pred
    
    def get_measures(self, model,test_x, test_y, index2label, label_type):    
        test_pred = self.predict_test(model, test_x, index2label )        
        test_y = np.argmax(test_y, -1)
        test_y = [[index2label[i] for i in sent] for sent in test_y ]
        
        if label_type == "cues": true_labels = CUES
        elif label_type == "cues_spec": true_labels = CUES_SPEC
        elif label_type == "scopes": true_labels = SCOPES
        elif label_type == "events": true_labels = EVENTS
            
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(test_y)):
            for j in range(len(test_y[i])):
                if test_y[i][j]!= "PAD": 
                    if test_y[i][j] in true_labels:  
                        if test_pred[i][j] == test_y[i][j]:
                            tp = tp+1
                        else:
                            fn = fn+1
                    else:  # No Cue/ Out of scope case 'N_C'
                        if test_pred[i][j] == test_y[i][j]:
                            tn = tn+1
                        else:
                            fp = fp+1
                            
                else:break
            
        
        cm ={"tp":tp, "tn":tn, "fp":fp, "fn":fn}
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        mesures = {"cm":cm, "precision":precision, "recall":recall, "f1": f1}
        return mesures
    
    def predict_single(self, model, features_dict, dp_obj, tr_obj, sent_tuple, detail_dict, max_len, index_dict, token_dict_scope, index2label, isIncludeNonCue, isLower):
        sent_dic = {}
        sent_dic[sent_tuple] = detail_dict[sent_tuple]
        te_data = dp_obj.data_for_scope_resolution(sent_dic, isIncludeNonCue)
        te_proc_data = tr_obj.get_processed_data(max_len, index_dict, token_dict_scope, te_data, isLower)
        
        test_x, _, _ = tr_obj.prepare_training_data(te_proc_data, features_dict, index_dict)
        
        pred = self.predict_test(model, test_x, index2label)
        return pred 
    
    def predict_single_elmo(self, model, features_dict, dp_obj, tr_obj, sent_tuple, detail_dict, max_len, index_dict, token_dict_scope, index2label, isIncludeNonCue, isLower):
        sent_dic = {}
        sent_dic[sent_tuple] = detail_dict[sent_tuple]
        te_data = dp_obj.data_for_scope_resolution(sent_dic, isIncludeNonCue)
        te_proc_data = tr_obj.get_processed_data(max_len, index_dict, token_dict_scope, te_data, isLower)
        
        test_x, _, _ = tr_obj.prepare_training_data_elmo(te_proc_data, features_dict, index_dict)
        
        pred = self.predict_test(model, test_x, index2label)
        return pred 
    
    def tag_negation_scopes(self, model, features_dict, dp_obj, tr_obj, obj_list, detail_dict, max_len, index_dict, token_dict_scope, index2label, isIncludeNonCue, isLower):
        flag = True;
        negation_dict = {}    
        
        for i in range(len(obj_list) ):
            uniqe_tuple = (obj_list[i].chap_name, obj_list[i].sent_num, obj_list[i].token_num)
            
            if flag == True:
                sent_tuple = (obj_list[i].chap_name, obj_list[i].sent_num)
                num_cues = detail_dict[sent_tuple][1] # index 1 stores number of cues
                if num_cues > 0:
                    pred_sent = self.predict_single(model, features_dict, dp_obj, tr_obj, sent_tuple, detail_dict, max_len, index_dict, token_dict_scope, index2label, isIncludeNonCue, isLower)
                j = 0                     
                flag = False;
                
            if num_cues == 0:
                negation_dict[uniqe_tuple] = []
            else:
                negation_list = obj_list[i].negation_list
                for k in range(num_cues):
                    if pred_sent[k][j] == "I_S":
                        if negation_list[k][1] == "_":
                            negation_list[k] = (negation_list[k][0], obj_list[i].word, negation_list[k][2])
                
                negation_dict[uniqe_tuple] = negation_list
            j = j + 1 #to access elements in prediction list
                    
            if i+1 < len(obj_list) and int(obj_list[i+1].token_num) == 0:
                flag = True
                
        return negation_dict
    
    def tag_negation_scopes_elmo(self, model, features_dict, dp_obj, tr_obj, obj_list, detail_dict, max_len, index_dict, token_dict_scope, index2label, isIncludeNonCue, isLower):
        flag = True;
        negation_dict = {}    
        
        for i in range(len(obj_list) ):
            uniqe_tuple = (obj_list[i].chap_name, obj_list[i].sent_num, obj_list[i].token_num)
            
            if flag == True:
                sent_tuple = (obj_list[i].chap_name, obj_list[i].sent_num)
                num_cues = detail_dict[sent_tuple][1] # index 1 stores number of cues
                if num_cues > 0:
                    pred_sent = self.predict_single_elmo(model, features_dict, dp_obj, tr_obj, sent_tuple, detail_dict, max_len, index_dict, token_dict_scope, index2label, isIncludeNonCue, isLower)
                j = 0                     
                flag = False;
                
            if num_cues == 0:
                negation_dict[uniqe_tuple] = []
            else:
                negation_list = obj_list[i].negation_list
                for k in range(num_cues):
                    if pred_sent[k][j] == "I_S":
                        if negation_list[k][1] == "_":
                            negation_list[k] = (negation_list[k][0], obj_list[i].word, negation_list[k][2])
                
                negation_dict[uniqe_tuple] = negation_list
            j = j + 1 #to access elements in prediction list
                    
            if i+1 < len(obj_list) and int(obj_list[i+1].token_num) == 0:
                flag = True
                
        return negation_dict
    
    def tag_negation_events(self, model, features_dict, dp_obj, tr_obj, obj_list, detail_dict, max_len, index_dict_scope, token_dict_scope, index2label, isIncludeNonCue, isLower):
        flag = True;
        negation_dict = {}    
        
        for i in range(len(obj_list) ):
            uniqe_tuple = (obj_list[i].chap_name, obj_list[i].sent_num, obj_list[i].token_num)
            
            if flag == True:
                sent_tuple = (obj_list[i].chap_name, obj_list[i].sent_num)
                num_cues = detail_dict[sent_tuple][1] # index 1 stores number of cues
                if num_cues > 0:
                    pred_sent = self.predict_single(model, features_dict, dp_obj, tr_obj, sent_tuple, detail_dict, max_len, index_dict_scope, token_dict_scope, index2label, isIncludeNonCue, isLower)
                j = 0                     
                flag = False;
                
            if num_cues == 0:
                negation_dict[uniqe_tuple] = []
            else:
                negation_list = obj_list[i].negation_list
                for k in range(num_cues):
                    if pred_sent[k][j] == "I_E":
                        if negation_list[k][2] == "_":
                            negation_list[k] = (negation_list[k][0], negation_list[k][1], obj_list[i].word)
                
                negation_dict[uniqe_tuple] = negation_list
            j = j + 1 #to access elements in prediction list
                    
            if i+1 < len(obj_list) and int(obj_list[i+1].token_num) == 0:
                flag = True
                
        return negation_dict
    
    def tag_negation_events2(self, model, features_dict, dp_obj, tr_obj, obj_list, detail_dict, max_len, index_dict_scope, token_dict_scope, index2label, isIncludeNonCue, isLower):
        flag = True;
        negation_dict = {}    
        
        for i in range(len(obj_list) ):
            uniqe_tuple = (obj_list[i].chap_name, obj_list[i].sent_num, obj_list[i].token_num)
            
            if flag == True:
                sent_tuple = (obj_list[i].chap_name, obj_list[i].sent_num)
                num_cues = detail_dict[sent_tuple][1] # index 1 stores number of cues
                if num_cues > 0:
                    pred_sent = self.predict_single(model, features_dict, dp_obj, tr_obj, sent_tuple, detail_dict, max_len, index_dict_scope, token_dict_scope, index2label, isIncludeNonCue, isLower)
                j = 0                     
                flag = False;
                
            if num_cues == 0:
                negation_dict[uniqe_tuple] = []
            else:
                negation_list = obj_list[i].negation_list
                for k in range(num_cues):
                    if pred_sent[k][j] == "I_E":
                        if negation_list[k][1] != "_":
                            negation_list[k] = (negation_list[k][0], negation_list[k][1], obj_list[i].word)
                
                negation_dict[uniqe_tuple] = negation_list
            j = j + 1 #to access elements in prediction list
                    
            if i+1 < len(obj_list) and int(obj_list[i+1].token_num) == 0:
                flag = True
                
        return negation_dict
    
    def get_num_cues(self, pred_sent):
        num_cues = 0
        visited = False
        for i in range(len(pred_sent)):
            if pred_sent[i] in ('S_C', 'PRE_C', 'POST_C'):
                num_cues = num_cues+1
            elif pred_sent[i] == "M_C" and visited == False:
                num_cues = num_cues+1
                visited = True
        return num_cues
    
    def get_prefix_cue(self, word):
        global global_pre_miss
        for prefix in PREFIX_CUES:
            position = word.find(prefix, 0, len(prefix))
            if position >= 0:
                return prefix, word[position + len(prefix) : len(word)]
        
        global_pre_miss = global_pre_miss+1
        return word, "_"

    def get_suffix_cue(self, word):
        global global_post_miss
        for suffix in SUFFIX_CUES:
            position = word.find(suffix)
            if position >= 0:
                return suffix, word[0:position]
        
        global_post_miss = global_post_miss+1            
        return word, "_"
    
    def tag_negation_cues(self, obj_list, prediction):
        flag = True; ii = 0
        negation_dict = {}    
        for i in range(len(obj_list) ):
            uniqe_tuple = (obj_list[i].chap_name, obj_list[i].sent_num, obj_list[i].token_num)
            if flag == True:
                pred_sent = prediction[ii]; j = 0
                num_cues = self.get_num_cues(pred_sent)          
                flag = False; k = 0; multi_k = -1
            if num_cues == 0:
                negation_dict[uniqe_tuple] = []
            else:
                negation_list = [("_", "_", "_") for n in range(num_cues)]
                #print ("i = {}, ii= {}, pred_sent len: {}, j = {}, k = {}".format(i,ii, len(pred_sent), j, k))
                if j < len(pred_sent):
                    if pred_sent[j] == 'S_C':
                        if k == multi_k: k = k+1
                        negation_list[k] = (obj_list[i].word, "_", "_")
                        k = k+1
                    elif pred_sent[j] == 'PRE_C':
                        if k == multi_k: k = k+1
                        cue, scope = self.get_prefix_cue(obj_list[i].word.lower())
                        negation_list[k] = (cue, scope, "_") 
                        k = k+1
                    elif pred_sent[j] == 'POST_C':
                        if k == multi_k: k = k+1
                        cue, scope = self.get_suffix_cue(obj_list[i].word.lower())
                        negation_list[k] = (cue, scope, "_") 
                        k = k+1
                    elif pred_sent[j] == 'M_C':
                        if multi_k == -1: multi_k = k
                        negation_list[multi_k] = (obj_list[i].word, "_", "_")
                
                negation_dict[uniqe_tuple] = negation_list
            j = j + 1 #to access elements in prediction list
                    
            if i+1 < len(obj_list) and int(obj_list[i+1].token_num) == 0:
                flag = True; k = 0; multi_k = -1; ii = ii+1; j = 0

        return negation_dict



class plotting():
    def plot_accuracy(self, history, file_name, file_format, dpi_=1200):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy', fontsize=14, color='black')
        plt.xlabel('Epoch', fontsize=14, color='black')
        plt.legend(['train', 'validation'], loc='upper left')
        #plt.savefig(file_name, format=file_format, dpi=dpi_)
        plt.savefig(file_name, format=file_format)
        plt.close()
        
    def plot_loss(self, history, file_name, file_format, dpi_=1200):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss', fontsize=14, color='black')
        plt.xlabel('Epoch', fontsize=14, color='black')
        plt.legend(['train', 'validation'], loc='upper left')
        #plt.savefig(file_name, format=file_format, dpi=dpi_)
        plt.savefig(file_name, format=file_format)
        plt.close()
        
    