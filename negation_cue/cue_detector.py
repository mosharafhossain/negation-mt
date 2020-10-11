# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import negation_cue.data_prep as data_prep
import negation_cue.embeddings as embeddings
import negation_cue.nn as nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import negation_cue.evaluation as evaluation
import spacy
from collections import defaultdict
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


class train():
    def __init__(self, params):
        
        # Parameter Settings
        self.max_len           = params["max_len"]     # maximum allowed length of sentences
        self.batch_size        = params["batch_size"]
        self.training_file     = params["training_file"]
        self.embed_size        = params["embed_size"]  # embedding dimension
        self.embed_file        = params["embed_file"]
        self.features_dict     = params["features_dict"]
        self.num_epoch         = params["num_epoch"]
        self.model_path        = params["model_path"]
        self.final_model_path  = params["final_model_path"]
        self.patience          = params["patience"]
        self.model_type        = params["model_type"]
        
        # Network params
        self.cell_units        = params["cell_units"]   
        self.drpout            = params["drpout"] 
        self.rec_drpout        = params["rec_drpout"] 
        self.validation_split  = params["validation_split"] 
        
        self.vrbose            = params["vrbose"]        
        self.cue_tr_obj = data_prep.data_for_training_cue()
        
        
        
    def prepare_tr_data(self):
        """
        Prepare training data.
        """
        training_obj = open(self.training_file, "r", encoding="utf8") #training + development        
        tr_refined_data, token_dict, index_dict = self.cue_tr_obj.get_data_for_training(training_obj, self.max_len)
        return tr_refined_data, token_dict, index_dict
    
    def prepare_embed_matrix(self, index_dict):
        """
        Prepare Embedding matrix.
        """
        word2index    = index_dict["word2index"]
        embed_obj     = embeddings.word_embed(self.max_len, self.embed_size, self.batch_size)
        embed_matrix  = embed_obj.glove_embedding(self.embed_file, self.embed_size, word2index)
        return embed_matrix


    def fit(self, isTrain):
        """
        Fit a neural model.
        """        
        tr_refined_data, token_dict, index_dict = self.prepare_tr_data()
        embed_matrix = self.prepare_embed_matrix(index_dict)
        
        train_y = np.array(tr_refined_data["labels"])
        train_x, num_tokens, embed_dims = self.cue_tr_obj.prepare_training_data(tr_refined_data, self.features_dict, index_dict, self.embed_size)
        num_labels = len(index_dict["tag2index"]) # This includes PAD tag
        model_obj = nn.cue_models() #used as cue model here. #model_obj = models.cue_models()
        
        
        # Model building and Training___________________        
        

        if self.model_type == "BiLSTM_CRF":
            crf, model = model_obj.bilstm_glove_crf(self.max_len, self.features_dict, num_tokens, num_labels, embed_dims,  embed_matrix, self.cell_units, self.drpout, self.rec_drpout)
            early_stop = EarlyStopping(monitor='val_crf_viterbi_accuracy', mode='max', patience=self.patience)  #or monitor='val_loss', mode='min', Training stops after 5/7(patience) consecutive epochs if no improvement in the validation loss.
            model_check = ModelCheckpoint(self.model_path, monitor='val_crf_viterbi_accuracy', mode='max', save_best_only=True, verbose=self.vrbose) #or monitor='val_loss', mode='min'
        else: #model without CRF
            model = model_obj.bilstm_glove(self.max_len, self.features_dict, num_tokens, num_labels, embed_dims,  embed_matrix, self.cell_units, self.drpout, self.rec_drpout)
            early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)  
            model_check = ModelCheckpoint(self.model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=self.vrbose) #or monitor='val_loss', mode='min'
            crf   = None
            
        if isTrain: # For the model training phase           
            if self.model_type == "BiLSTM_CRF": 
                #model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy]) #default learning rate 0.001. Link:https://keras.io/optimizers/
                model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_viterbi_accuracy]) #default learning rate 0.001. Link:https://keras.io/optimizers/
            else:
                model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        
            history = model.fit(train_x, train_y, \
                                batch_size=self.batch_size, \
                                epochs=self.num_epoch, shuffle=True, \
                                validation_split=self.validation_split,
                                callbacks=[early_stop, model_check], \
                                verbose=self.vrbose)
            
            #Load the best model
            model.load_weights(self.model_path) #load the weights of the best model
            
            # Save best model
            model.save(self.final_model_path)
            output_dict = {"history":history, "model": model, "crf":crf, "token_dict":token_dict, "index_dict":index_dict, "max_len":self.max_len, "features_dict":self.features_dict}
        else: # if the model is loaded from a saved file              
            model = load_model(self.final_model_path) # load the trained model
            output_dict = {"model": model, "crf":crf, "token_dict":token_dict, "index_dict":index_dict, "max_len":self.max_len, "features_dict":self.features_dict}
        
        return output_dict
    
    def evaluate_model(self, test_file, output_dict, output_file):
        """
        This method generate an output file by predicting a data in a given file with the trained model.
        """
        test_file  = open(test_file, "r", encoding="utf8")
        index_dict = output_dict["index_dict"]
        token_dict = output_dict["token_dict"]
        model      = output_dict["model"]
        te_refined_data, test_data_objs = self.cue_tr_obj.get_data_for_validation(test_file, self.max_len, index_dict, token_dict)
                
        # Prediction        
        index2tag    = index_dict["index2tag"] 
        test_x, _, _ = self.cue_tr_obj.prepare_training_data(te_refined_data, self.features_dict, index_dict, self.embed_size)
        
        eval_obj = evaluation.evaluation()
        test_pred = eval_obj.predict_test(model, test_x, index2tag)
        #print("Unique labels: {}".format(np.unique(test_pred) ))
        
        
        # Create new file
        negation_dict = eval_obj.tag_negation_cues(test_data_objs, test_pred)
        dp_obj = data_prep.data_preparation()
        new_obj_list = dp_obj.create_new_obj_list(test_data_objs, negation_dict)
        dp_obj.print_to_file(new_obj_list, output_file)  # test_gold_pred_cue3 is the best
        
        
class cue_prediction():    
    def prepare_data(self, orig_sent_list, model_dict):
        """
        prepare data for training.
        params:
            @orig_sent_list: (list), list of all original/raw sentences/segments.
            @model_dict: (dictionary), contains information regarding the pre-trained model
        output:
            @x: (list): data is formatted based on model input.
        """
        # Collect information
        nlp = spacy.load("en_core_web_sm")
        obj = data_prep.data_for_training_cue()        
        index_dict     = model_dict["index_dict"]
        token_dict     = model_dict["token_dict"]
        features_dict  = model_dict["features_dict"]
        max_len        = model_dict["max_len"]
        
        
        # Tokenize the sentences
        sent_list = []
        upos_list = []
        for sent in orig_sent_list:
            spacy_doc = nlp(sent)
            word_tok_list   = [token.text for token in spacy_doc] # tokens of the sentence
            upos_tok_list   = [token.pos_ for token in spacy_doc] # universal POS of the sentence
            
            sent_list.append(word_tok_list)
            upos_list.append(upos_tok_list)
        
        
        # for sentences
        pad_value = index_dict["word2index"]["PAD"]
        sentences = obj.indexing_with_padding(sent_list, token_dict["words"], index_dict["word2index"], max_len, pad_value)
        
        # For Universal POS
        pad_value = index_dict["upos2index"]["PAD"]
        upos = obj.indexing_with_padding(upos_list, token_dict["upos"], index_dict["upos2index"], max_len, pad_value)
        
        
        train_data = []
        if "words" in features_dict: 
            train_data.append(sentences)       
        if "upos" in features_dict: 
            train_data.append(upos)
        return train_data, sent_list
    
    def pred_info_sent(self, sent_tokens, pred_sent, cues = ["S_C", "PRE_C", "POST_C", "M_C"]):
        """
        This function generates information of a given sentence and a predicted outcome of that sentence.
        params:
            @pred_sent: (list), token-level prediction of a given sentence
            @sent_tokens: (list), tokens of a sentence.
        """
        num_cues = len( set(pred_sent).intersection(set(cues)) )  #no cue tag = "NA"
        neg_status = "w_neg" if num_cues > 0 else "wo_neg"
        tokens_cues = []  
        pred_len = len(pred_sent)
        
        for i in range(len(sent_tokens)):       
            if i < pred_len and pred_sent[i] in cues:tag = "Y"
            else: tag = "N"
            tokens_cues.append(sent_tokens[i]+"/"+tag) 
        tokens_cues = " ".join(tokens_cues) #making string. e.g "John/N does/N not/Y like/N soccer/N ./N"
        return tokens_cues, neg_status
    
    def get_cue_tags(self, cue_info_dict):
        """
        Tag a token if predicted as negation cue.
        params:
            @cue_info_dict: (dictionary), contains cue information 
           
        """
        line_number      = []
        neg_status       = []   
        tokens_cues_list = []
        sent_size = len(cue_info_dict["sent_tokens"])
        for i in range(sent_size):
            line_number.append(i+1)            
            pred_sent   = cue_info_dict["pred"][i]            
            sent_tokens = cue_info_dict["sent_tokens"][i]
            #cross check
            tokens_cues, status_1 = self.pred_info_sent(sent_tokens, pred_sent)
            tokens_cues_list.append(tokens_cues)        
            
            if i in cue_info_dict["indices"]["neg_indices"]: status_2 = "w_neg"    
            else: status_2 = "wo_neg"   
            
            #cross check
            assert status_1 == status_2
            neg_status.append(status_1)
        
        return tokens_cues_list      
       

    def cue_info_by_detector(self, pred, sent_list, isPrint=True):
        """
        This function gives information of cues of a list that has all the sentences.
        params:
            @pred: (list of list), predictions of list of list type. Inner list is token level, and outer list is sentence level.
            @sent_list: (list of list), list of all tokenized (by spacy) sentences
        Outputs: 
            @cue_info_dict: Contains cues information
        """
        num_sent = len(pred)        
        
        
        cues_count_dict    = {}
        cues_dict          = defaultdict(int)
        neg_indices        = []
        noneg_indices      = []
        
        singleword_cues    = 0 #single word cues
        prefix_cues        = 0 #prefix cues
        suffix_cues        = 0 #suffix cues
        multiwords_cues    = 0 #mutiword cues, e.g neither nor, by no means
        multi_cue_count    = 0
        single_cues_sent   = 0 # number of sentences having only one cues in per sentence
        multiple_cues_sent = 0 # number of sentences having more than one cues in per sentence.
        total_cues_count   = 0
        
        
        pred_all_sent= [] #store prediction excluding padding
        
        
        for i in range(num_sent):            
            multi_cue_count = 0    
            multi_words = ""
            pred_sent    = []
            for j in range(len(pred[i])):            
                if pred[i][j] == "S_C":
                    singleword_cues += 1
                    multi_cue_count += 1
                    cues_dict[sent_list[i][j].lower()] += 1
                elif pred[i][j] == "PRE_C":
                    prefix_cues += 1
                    multi_cue_count += 1
                    cues_dict[sent_list[i][j].lower()] += 1
                elif pred[i][j] == "POST_C":
                    suffix_cues += 1
                    multi_cue_count += 1
                    cues_dict[sent_list[i][j].lower()] += 1
                elif pred[i][j] == "M_C":                    
                    multi_words += sent_list[i][j].lower()+ " "
                    
                # Store prediction for individual sentence
                if pred[i][j] != "PAD":
                    pred_sent.append(pred[i][j])
                    
            if len(multi_words)>0:  
                cues_dict[multi_words.strip()] += 1
                multiwords_cues += 1 
                multi_cue_count += 1
            
            if multi_cue_count > 1:
                multiple_cues_sent += 1
            elif multi_cue_count == 1:
                single_cues_sent += 1
            total_cues_count += multi_cue_count
            
            
            # neg and noneg indices
            if multi_cue_count >= 1:
                neg_indices.append(i)
            else:
                noneg_indices.append(i)
                
            # Store prediction output without the "PAD" tokens for all sentences
            pred_all_sent.append(pred_sent)
            
        indices = {"neg_indices":neg_indices, "noneg_indices":noneg_indices}    
        cues_count_dict = {"singleword_cues":singleword_cues, "prefix_cues":prefix_cues, "suffix_cues":suffix_cues, "multiwords_cues":multiwords_cues, "single_cues_sent":single_cues_sent, "multiple_cues_sent":multiple_cues_sent}
        cue_info_dict = {"cues_count_dict":cues_count_dict, "cues_dict":cues_dict, "indices":indices, "pred": pred_all_sent, "sent_tokens": sent_list}
        
        return cue_info_dict
    
    def generate_cue_info(self, orig_sent_list, model_dict):
        """        
        params:
            @orig_sent_list: (list), list of all original/raw sentences/segments.
            @model_dict: (dictionary), contains information regarding the pre-trained model
        output:
            @cue_info_dict: (dictionary), cue information.
                            cue_info_dict["cues_count_dict"]: Cues count information
                            cue_info_dict["cues_dict"]: Cues and their counts
                            cue_info_dict["indices"]: indices of negated sentences and indices of non negated sentences            
        """
        
        eval_obj    = evaluation.evaluation()         
        model       = model_dict["model"] 
        index2tag   = model_dict["index_dict"]["index2tag"]
        
        train_data, sent_list   = self.prepare_data(orig_sent_list, model_dict) #dim: dictionary
        pred_sent               = eval_obj.predict_test(model, train_data, index2tag) # dim: list of list
        cue_info_dict           = self.cue_info_by_detector(pred_sent, sent_list, isPrint=True)
        return cue_info_dict
    
    