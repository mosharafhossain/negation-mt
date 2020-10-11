# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import tensorflow as tf
from keras import backend as K

class word_embed():
    def __init__(self, max_len, embed_size, batch_size, elmo=None):
        self.max_len = max_len
        self.embed_size = embed_size
        self.batch_size = batch_size        
        if elmo != None: self.elmo = elmo
        
    def glove_embedding(self, file_path, embed_size, word2index):        
        # creating a dictionary from file
        file_obj = open(file_path, "r", encoding="utf8") 
        embeddings_dic = {}
        for line in file_obj:
            splitted_values = line.split()
            word = splitted_values[0]
            coefficients = np.asarray(splitted_values[1:], dtype='float32')
            embeddings_dic[word] = coefficients
            
        # generating the embedding matrix    
        embedding_matrix = np.random.random((len(word2index), embed_size))   #len(word2index) = number of unique words + 2(padding + unknown)
        for word, i in word2index.items():
            embedding_vector = embeddings_dic.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
        return embedding_matrix
    
    def elmo_embedding(self, x):
        return self.elmo(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(self.batch_size*[self.max_len])
                          },
                          signature="tokens",
                          as_dict=True)["elmo"]
        
    def elmo_embedding2(self, x):
        return self.elmo(inputs=K.cast(x, tf.string),
                          signature="default",
                          as_dict=True)["elmo"]
        
