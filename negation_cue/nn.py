# -*- coding: utf-8 -*-

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
from keras_contrib.layers import CRF


class cue_models():    
    def bilstm_glove_crf(self, max_len, features_dict, num_tokens, num_labels, embed_dims,  embed_matrix, cell_units, drpout, rec_drpout):
        
        inputs = []
        if "words" in features_dict:        
            input_ = Input(shape=(max_len,))
            z = Embedding(input_dim=num_tokens["words"], output_dim=embed_dims["words"], input_length=max_len, trainable=False, weights=[embed_matrix])(input_) 
            inputs.append(input_)
            
        if "cues" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["cues"], output_dim=embed_dims["cues"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
            
        if "cues_spec" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["cues_spec"], output_dim=embed_dims["cues_spec"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
            
            
        if "pos" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["pos"], output_dim=embed_dims["pos"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
            
        if "upos" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["upos"], output_dim=embed_dims["upos"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])     
            
        if "syntax" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["syntax"], output_dim=embed_dims["syntax"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
            
        if "sdep" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["sdep"], output_dim=embed_dims["sdep"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
                                
        z = Bidirectional(LSTM(units=cell_units, return_sequences=True, dropout=drpout, recurrent_dropout=rec_drpout))(z)    
        z = Bidirectional(LSTM(units=cell_units, return_sequences=True, dropout=drpout, recurrent_dropout=rec_drpout))(z) 
                
        z = TimeDistributed(Dense(cell_units, activation="relu"))(z)
        crf = CRF(num_labels)
        z = crf(z)
        model = Model(inputs=inputs , outputs=z)
        return crf, model
    
    
    def bilstm_glove(self, max_len, features_dict, num_tokens, num_labels, embed_dims,  embed_matrix, cell_units, drpout, rec_drpout):
        
        inputs = []
        if "words" in features_dict:        
            input_ = Input(shape=(max_len,))
            z = Embedding(input_dim=num_tokens["words"], output_dim=embed_dims["words"], input_length=max_len, trainable=False, weights=[embed_matrix])(input_) 
            inputs.append(input_)
            
        if "cues" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["cues"], output_dim=embed_dims["cues"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
            
        if "cues_spec" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["cues_spec"], output_dim=embed_dims["cues_spec"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
            
            
        if "pos" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["pos"], output_dim=embed_dims["pos"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
            
        if "upos" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["upos"], output_dim=embed_dims["upos"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])     
            
        if "syntax" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["syntax"], output_dim=embed_dims["syntax"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
            
        if "sdep" in features_dict: 
            input_ = Input(shape=(max_len,))
            x = Embedding(input_dim=num_tokens["sdep"], output_dim=embed_dims["sdep"], input_length=max_len, trainable=True)(input_) 
            inputs.append(input_)
            z = concatenate([z, x])
                                
        z = Bidirectional(LSTM(units=cell_units, return_sequences=True, dropout=drpout, recurrent_dropout=rec_drpout))(z)    
        z = Bidirectional(LSTM(units=cell_units, return_sequences=True, dropout=drpout, recurrent_dropout=rec_drpout))(z) 
                
        z = TimeDistributed(Dense(num_labels, activation="softmax"))(z)  
        model = Model(inputs=inputs , outputs=z)
        return model