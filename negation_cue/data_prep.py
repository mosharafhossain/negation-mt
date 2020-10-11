import numpy as np
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import copy
import spacy
nlp = spacy.load("en_core_web_sm")


cues_prefix = ['dis', 'im', 'in', 'ir', 'un' ]
cues_postfix = ['less', 'lessly', 'lessness']
cues_multiword = ['by no means', 'on the contrary', 'rather than', 'neither nor' ]
cues_single_word = ['absence', 'except', 'fail', 'failed', 'neglect', 'neglected', 'never', 'no', 'none', 'nor', 'nobody', 'not', "n't", 'nothing', 'nowhere', 'prevent', 'refuse', 'refused', 'without' ]
cues_dict = {"single": cues_single_word, "multi": cues_multiword, "prefix": cues_prefix, "postfix": cues_postfix}
PREFIX_CUES = ['dis', 'im', 'in', 'ir', 'un' ]
SUFFIX_CUES = ['less', 'lessly', 'lessness']
PAD = "PAD"
UNKNOWN = "UNK"

class data_structure:
    def __init__(self, line):
        tokens = line.split()
        self.chap_name = tokens[0]
        self.sent_num = tokens[1]
        self.token_num = tokens[2]
        self.word = tokens[3]
        self.lemma = tokens[4]
        self.pos = tokens[5]
        self.syntax = tokens[6]
        self.negation_list = []
        
        if tokens[7] != "***":
            neg_tokens = tokens[7:]
            l = len(neg_tokens)
            i = 0
            while i < l:
                t = (neg_tokens[i], neg_tokens[i+1], neg_tokens[i+2])                
                self.negation_list.append(t)
                i = i+3   


class data_preparation:
    def data_load(self, file):
        list_objs = []
        for line in file:
            if len(line) >= 8:
                obj = data_structure(line)
                list_objs.append(obj)
        file.close()                
        return list_objs
    
    def get_sentences(self, list_objs):
        sentences = []
        chap_name = list_objs[0].chap_name
        sent_num = list_objs[0].sent_num
        sentence = [ list_objs[0].word ]
        
        for i in range(1, len(list_objs)):
            if (list_objs[i].chap_name == chap_name) and (list_objs[i].sent_num == sent_num):
                sentence.append(list_objs[i].word)
                chap_name = list_objs[i].chap_name
                sent_num = list_objs[i].sent_num  
                if i == len(list_objs)-1:
                    sentences.append(sentence)
            else:
                sentences.append(sentence)
                chap_name = list_objs[i].chap_name
                sent_num = list_objs[i].sent_num
                sentence = [ list_objs[i].word ] 
        return sentences
    
    def get_cue_tagging(self, list_objs, start_index, end_index, cues_position):
        sentence, cues_sent, pos_sent  = [], [], []
        
        for i in list(range(start_index, end_index+1)):
            sentence.append(list_objs[i].word)
            pos_sent.append(list_objs[i].pos)
            cues_sent.append("N_C")  # Initially tagging No Cue
                    
        if cues_position:            
            for num, cues_list in cues_position.items():
                if len(cues_list) == 1:
                    cue = cues_list[0][0]
                    position = cues_list[0][1] 
                    if cue in cues_dict["prefix"]:
                        cues_sent[position] = "PRE_C"  #Prefix Cue
                    elif cue in cues_dict["postfix"]:
                        cues_sent[position] = "POST_C" #Postfix Cue
                    else:
                        cues_sent[position] = "S_C" #Single word Cue
                else:
                    for item in cues_list:
                        position = item[1]
                        cues_sent[position] = "M_C" #Multiword Cue
                        
        return sentence, cues_sent, pos_sent
                
        
    
    def tag_all_cues(self, list_objs):
        sentence_list, pos_sent_list, cues_sent_list = [], [], []
        cues_position = defaultdict(list)
        flag = True
        for i in range(len(list_objs)):            
            if flag:
                start_indx = i
                flag = False
            
            neg_list = list_objs[i].negation_list
            if len(neg_list) > 0:
                for j in range( len(neg_list) ):
                    if neg_list[j][0] != "_":
                        cues_position[j].append( (neg_list[j][0], int(list_objs[i].token_num))  )
            
            if (i+1 < len(list_objs) and int(list_objs[i+1].token_num) == 0) or (i == len(list_objs)-1 ):   
                sentence, cues_sent, pos_sent = self.get_cue_tagging(list_objs, start_indx, i, cues_position)
                sentence_list.append(sentence)  
                cues_sent_list.append(cues_sent)
                pos_sent_list.append(pos_sent)                
                cues_position = defaultdict(list)
                flag = True
            
        #universal POS tagging using spaCy
        univ_pos_sent_list = []
        syntactic_dep_list = []
        num_child_list = []
        for sent in sentence_list:
            sent = " ".join(sent)
            sent = nlp(sent)
            upos_tags = [token.pos_ for token in sent]
            synt_dep  = [token.dep_ for token in sent]
            num_child = [len([child for child in token.children]) for token in sent]
            
            univ_pos_sent_list.append(upos_tags)
            syntactic_dep_list.append(synt_dep)
            num_child_list.append(num_child)
                    
        data = {"sentences": sentence_list, "pos": pos_sent_list, "univ_pos":univ_pos_sent_list, "syn_dep":syntactic_dep_list, "syn_child":num_child_list,"cues":cues_sent_list}        
        return data

            
    def print_method(self, list_objs, count = 50, isAll= False):
        line = ''
        delim = "\t"
        for i in range(len(list_objs)):
            line = list_objs[i].chap_name + delim + list_objs[i].sent_num + delim +list_objs[i].token_num +  delim + list_objs[i].word + delim + list_objs[i].lemma + delim + list_objs[i].pos + delim + list_objs[i].syntax
            if len(list_objs[i].negation_list) == 0:
                line = line + delim + "***"
            else:
                for elem in list_objs[i].negation_list:
                     line = line + delim + elem[0] + delim + elem[1]+ delim + elem[2]
            
            if isAll == False:
                print(line)
                if i == count:
                    break
            else:
                print(line)
            
            #print (i+1, len(list_objs),  list_objs[i].token_num, list_objs[i+1].token_num)
            if i+1 < len(list_objs)  and int(list_objs[i+1].token_num) == 0:                
                print("")
                
    def print_to_file(self, list_objs, file_name):
        file_obj = open(file_name, "w")
        line = ''
        delim = "\t"
        for i in range(len(list_objs)):
            line = list_objs[i].chap_name + delim + list_objs[i].sent_num + delim +list_objs[i].token_num +  delim + list_objs[i].word + delim + list_objs[i].lemma + delim + list_objs[i].pos + delim + list_objs[i].syntax
            if len(list_objs[i].negation_list) == 0:
                line = line + delim + "***"
            else:
                for elem in list_objs[i].negation_list:
                     line = line + delim + elem[0] + delim + elem[1]+ delim + elem[2]
            
            file_obj.write(line)
            file_obj.write("\n")
            
            #print (i+1, len(list_objs),  list_objs[i].token_num, list_objs[i+1].token_num)
            if i+1 < len(list_objs)  and int(list_objs[i+1].token_num) == 0:                
                file_obj.write("\n")
        file_obj.close()
        
    # Returns universal PoS of the words in a sentence. Spacy is used here.
    def get_universal_pos(self, sentence):
        sentence = " ".join(sentence) #creating a sentence from a list of words
        sentence = nlp(sentence)
        upos = [token.pos_ for token in sentence]
        return upos
    

    
    def get_syntactical_info(self, sentence):
        sentence = " ".join(sentence) #creating a sentence from a list of words
        sentence = nlp(sentence)
        upos = [token.pos_ for token in sentence]
        synt_dep  = [token.dep_ for token in sentence]
        syn_child = [len([child for child in token.children]) for token in sentence]
        return upos, synt_dep, syn_child
    
    
    # Returns type of cue.
    def get_cue_type(self, start_index, cur_i, j, list_objs):            
        counter = 0        
        for i in range(start_index, len(list_objs)):
            if list_objs[i].negation_list[j][0] != "_":
                counter += 1
            if (i == len(list_objs)-1) or ( i+1 < len(list_objs) and int(list_objs[i+1].token_num) == 0): 
                break
        
        if counter > 1:
            return "M_C"
        else:
            cue = list_objs[cur_i].negation_list[j][0].lower()         
            word = list_objs[cur_i].word
            if cue in PREFIX_CUES and len(word) > len(cue):
                return "PRE_C"
            elif cue in SUFFIX_CUES and len(word) > len(cue):
                return "POST_C"
            else:
                return "S_C"
    
    # Prepares a dictionary of data includes all availavle and made-of features.    
    def get_data_details(self, list_objs):
        data_dict = {}

        isStart = True
        word_list   = []
        lemma_list  = []
        pos_list    = []
        syntax_list = []        
        cues_dict      = defaultdict(list)  # Store cue information for a sentence
        cues_spec_dict = defaultdict(list)  # Store specific type of the cue (single word, multi-word, prefix, suffix) 
        scope_dict     = defaultdict(list)  # Store Scope information 
        event_dict     = defaultdict(list)  # Store Negated event information
        #temp_cue_dict  = defaultdict(list)  # Store number of cues for a negation
        
        for i in range(len(list_objs)):
            
            # Sentence Start checking
            if isStart == True: 
                start_index = i
                isStart = False
            
            # Extract features information
            unique_tuple = (list_objs[i].chap_name, list_objs[i].sent_num)  
            word_list.append(list_objs[i].word)
            lemma_list.append(list_objs[i].lemma)
            pos_list.append(list_objs[i].pos)
            syntax_list.append(list_objs[i].syntax)
                                    
            # Extract Negation information
            neg_list = list_objs[i].negation_list
            num_cues = len(neg_list)            
            if num_cues > 0:
                for j in range( num_cues ):
                    if neg_list[j][0] != "_": cues_dict[j].append( "I_C"  )
                    else: cues_dict[j].append( "O_C"  )
                    
                    if neg_list[j][0] != "_": cues_spec_dict[j].append( self.get_cue_type( start_index, i, j, list_objs) )
                    else: cues_spec_dict[j].append( "O_C"  )
                    
                    if neg_list[j][1] != "_": scope_dict[j].append( "I_S" )
                    else: scope_dict[j].append( "O_S" )
                    
                    if neg_list[j][2] != "_": event_dict[j].append( "I_E" )
                    else: event_dict[j].append( "O_E" )                    
                                
            # Check for reaching last token of a sentence
            if (i == len(list_objs)-1) or ( i+1 < len(list_objs) and int(list_objs[i+1].token_num) == 0): 
                num_words = int(list_objs[i].token_num) + 1
                #upos_list = self.get_universal_pos(word_list) 
                upos_list, synt_dep_list, syn_child_list = self.get_syntactical_info(word_list)
                data_dict[unique_tuple] = [num_words, num_cues, word_list, lemma_list, pos_list, upos_list, syntax_list, cues_dict, cues_spec_dict, scope_dict, event_dict, synt_dep_list, syn_child_list]
                
                # Reset all variables to extract information for a new sentence
                isStart = True
                word_list   = []
                lemma_list  = []
                pos_list    = []
                syntax_list = []                
                cues_dict      = defaultdict(list)
                cues_spec_dict = defaultdict(list)
                scope_dict     = defaultdict(list)
                event_dict     = defaultdict(list)   
                #temp_cue_dict  = defaultdict(list)
        
        return data_dict
    
    # Returns data for cue detection. This data includes sentences whether or not the sentences have cues.
    def data_for_cue_resolution(self, detail_data_dict):
        data = defaultdict(list)
        for key, value in detail_data_dict.items():
            data["sentences"].append(value[2])      # index 2 stores list of words of a sentence
            data["lemma"].append(value[3])          # index 3 stores list of lemmas of the words of a sentence
            data["pos"].append(value[4])            # index 4 stores list of PoSs of a sentence
            data["upos"].append(value[5])       # index 5 stores list of universal PoSs of a sentence
            data["syntax"].append(value[6])         # index 6 stores list of syntax of the words of a sentence            
            
            num_words = value[0]  # index 0 stores number of words in a sentence
            num_cues  = value[1]   # index 1 stores number of cues in a sentence            

            default_cue_list = ["O_C" for _ in range(num_words)]   # Setting Not cue ('O_C') for all word positions in a sentence
            if num_cues > 0:
                for i in range(num_cues):  
                    cue_list = value[8][i] # index 8 stores dictionary of specific cues in a sentences
                    for j in range(num_words):
                        if cue_list[j] != "O_C": #if Cue (Not a Non Cue)
                            default_cue_list[j] = cue_list[j]
                        
                        
            data["cues_spec"].append(default_cue_list)   
                    
        return data
    
    
    # Incorporates sentences which have at least one cue.
    # Returning data includes all available and made-of features and labels. If a sentence contains multiple number of cues (btw, by no means is as a whole single cue, but multiple word cue), the
    # features (e.g sentences, lemma) copies that number of times.
    def data_for_scope_resolution(self, detail_data_dict, isIncludeNonCue = False):
        data = defaultdict(list)
        for key, value in detail_data_dict.items():
            num_cues = value[1]   # index 1 stores number of cues
            if num_cues > 0:
                for i in range(num_cues):  # does not include the data with no cues
                    data["sentences"].append(value[2])      # index 2 stores list of words of a sentence
                    data["lemma"].append(value[3])          # index 3 stores list of lemmas of the words of a sentence
                    data["pos"].append(value[4])            # index 4 stores list of PoSs of a sentence
                    data["upos"].append(value[5])           # index 5 stores list of universal PoSs of a sentence
                    data["syntax"].append(value[6])         # index 6 stores list of syntax of the words of a sentence
                    data["cues"].append(value[7][i])        # index 7 stores dictionary of cues in a sentences
                    data["cues_spec"].append(value[8][i])   # index 8 stores dictionary of specific cues in a sentences
                    data["scopes"].append(value[9][i])      # index 9 stores dictionary of scopes in a sentences
                    data["events"].append(value[10][i])     # index 10 stores dictionary of negated events in a sentences
                    data["sdep"].append(value[11])          # index 11 stores list of syntactic dependency in a sentences
                    data["schild"].append(value[12])        # index 12 stores list of syntactic children in a sentences
            else:
                if isIncludeNonCue == True:
                    num_words = value[0]                                       # index 0 stores number of words in a sentence
                    data["sentences"].append(value[2])                         # index 2 stores list of words of a sentence
                    data["lemma"].append(value[3])                             # index 3 stores list of lemmas of the words of a sentence
                    data["pos"].append(value[4])                               # index 4 stores list of PoSs of a sentence
                    data["upos"].append(value[5])                              # index 5 stores list of universal PoSs of a sentence
                    data["syntax"].append(value[6])                            # index 6 stores list of syntax of the words of a sentence
                    data["cues"].append(["O_C" for _ in range(num_words)])     # index 7 stores dictionary of cues in a sentences
                    data["cues_spec"].append(["O_C" for _ in range(num_words)])# index 8 stores dictionary of specific cues in a sentences
                    data["scopes"].append(["O_S" for _ in range(num_words)])   # index 9 stores dictionary of scopes in a sentences
                    data["events"].append(["O_E" for _ in range(num_words)])   # index 10 stores dictionary of negated events in a sentences
                    data["sdep"].append(value[11])                             # index 11 stores list of syntactic dependency in a sentences
                    data["schild"].append(value[12])                           # index 12 stores list of syntactic children in a sentences
        return data
    
    def create_new_obj_list(self, obj_list, negation_dict):
        newobj_list = copy.deepcopy(obj_list)
        for i in range(len(obj_list)):
            new_negation = negation_dict[(obj_list[i].chap_name, obj_list[i].sent_num, obj_list[i].token_num)]
            newobj_list[i].negation_list = new_negation
            
        return newobj_list
    
    def get_gold_cue_file(self, obj_list):
        newobj_list = copy.deepcopy(obj_list)
        for i in range(len(obj_list)):
            old_neg_list = obj_list[i].negation_list
            num_cues = len(old_neg_list)
            neg_list = []
            for nc in range(num_cues):
                if old_neg_list[nc][0] != "_": 
                    neg_list.append((old_neg_list[nc][0], "_", "_"))
                else: neg_list.append(("_", "_", "_"))
                        
            newobj_list[i].negation_list = neg_list            
        return newobj_list
    
    def get_gold_cue_file_pp(self, obj_list):
        newobj_list = copy.deepcopy(obj_list)
        for i in range(len(obj_list)):
            old_neg_list = obj_list[i].negation_list
            num_cues = len(old_neg_list)
            neg_list = []
            for nc in range(num_cues):
                cue = old_neg_list[nc][0] 
                word= obj_list[i].word.lower()
                if cue != "_": 
                    if cue.lower() in PREFIX_CUES:                        
                        position = word.find(cue, 0, len(cue))
                        scope = word[position + len(cue) : len(word)]
                        neg_list.append((cue, scope, "_"))                    
                    elif cue.lower() in SUFFIX_CUES:                        
                        position = word.find(cue)
                        scope = word[0:position]
                        neg_list.append((cue, scope, "_"))                        
                    else: neg_list.append((cue, "_", "_"))
                    
                else: neg_list.append(("_", "_", "_"))
                        
            newobj_list[i].negation_list = neg_list            
        return newobj_list

 
class data_for_training_cue():
    def get_unique_tokens(self, data):
        token_dict = {}     
        # For all words in the data
        words_all = set()
        for sentence in data["sentences"]:
            for word in sentence:
                words_all.add(word)
        token_dict["words"] = list(words_all)   
        
        # For all Universal POS in the data
        pos_all = set()
        for sentence in data["pos"]:
            for pos in sentence:
                pos_all.add(pos)
        token_dict["pos"] = list(pos_all)
        
        # For all Universal POS in the data
        upos_all = set()
        for sentence in data["univ_pos"]:
            for upos in sentence:
                upos_all.add(upos)
        token_dict["upos"] = list(upos_all)      
        
        # For all syntactic dependency in the data
        syn_dep_all = set()
        for sentence in data["syn_dep"]:
            for syn_dep in sentence:
                syn_dep_all.add(syn_dep)
        token_dict["sdep"] = list(syn_dep_all)      
        
        # For all Negation Cues in the data    
        tags_all = set()
        for sentence in data["cues"]:
            for tag in sentence:
                tags_all.add(tag)
        token_dict["tags"] = list(tags_all)  
        
        return token_dict
    
    
    def get_indexing(self, token_dict):
        word2index = {w:i+2 for i, w in enumerate(token_dict["words"] )}   
        word2index["PAD"] = 0
        word2index["UNK"] = 1 #for unknown word
        index2word = {i:w for w, i in word2index.items()}
            
        upos2index = {w:i+2 for i, w in enumerate(token_dict["upos"] )}   
        upos2index["PAD"] = 0
        upos2index["UNK"] = 1 #for unknown word
        index2upos = {i:w for w, i in upos2index.items()}
        
        sdep2index = {w:i+2 for i, w in enumerate(token_dict["sdep"] )}   
        sdep2index["PAD"] = 0 
        sdep2index["UNK"] = 1 #for unknown word
        index2sdep = {i:w for w, i in sdep2index.items()}
                
        tag2index = {w:i+1 for i, w in enumerate(token_dict["tags"] )} 
        tag2index["PAD"] = 0
        index2tag = {i:w for w, i in tag2index.items()}
        
        index_dict = {"word2index": word2index, "index2word": index2word, "upos2index": upos2index, "index2upos":index2upos, "sdep2index": sdep2index, "index2sdep":index2sdep, "tag2index": tag2index, "index2tag": index2tag }
        return index_dict
    
                
    
    def indexing_with_padding(self, sentence_list, unique_list, token2index, max_len, pad_value):
        indexed_sent_list = []
        for sentence in sentence_list:
            token_list = []
            for token in sentence:
                if token in unique_list:
                    token_list.append(token2index[token] )
                else:
                    token_list.append(token2index["UNK"])  #for unknown word
            indexed_sent_list.append(token_list)
            token_list = []
        
        indexed_sent_list = pad_sequences(maxlen=max_len, sequences=indexed_sent_list, padding="post", value=pad_value)
        return indexed_sent_list
        
    
    
    def one_hot_encoding(self, labels, num_labels):
        labels = [to_categorical(i, num_classes = num_labels + 1) for i in  labels] # num_labels + 1, 1 is added for padding
        return labels
        
    
    def get_refined_data(self, max_len, index_dict, token_dict, data):
        # For Sentences
        pad_value = index_dict["word2index"]["PAD"]
        sentences = self.indexing_with_padding(data["sentences"], token_dict["words"], index_dict["word2index"], max_len, pad_value)
        
        # For Universal POS
        pad_value = index_dict["upos2index"]["PAD"]
        upos = self.indexing_with_padding(data["univ_pos"], token_dict["upos"], index_dict["upos2index"], max_len, pad_value)
        
        # For Syntactic dependency
        pad_value = index_dict["sdep2index"]["PAD"]
        sdep = self.indexing_with_padding(data["syn_dep"], token_dict["sdep"], index_dict["sdep2index"], max_len, pad_value)
        
        # For tags 
        pad_value = index_dict["tag2index"]["PAD"]
        tags = self.indexing_with_padding(data["cues"], token_dict["tags"], index_dict["tag2index"], max_len, pad_value)
        
        # one hot encoding of labels
        unique_labels_size = len(token_dict["tags"])
        labels = self.one_hot_encoding(tags, unique_labels_size)
        
        refined_data = {"sentences": sentences, "upos": upos, "sdep": sdep, "labels": labels}
        return refined_data
    
    
    def get_data_for_training(self, file_name, max_len):
        dp_obj = data_preparation()
        tr_objs = dp_obj.data_load(file_name)
        tr_data = dp_obj.tag_all_cues(tr_objs)        
        
        token_dict = self.get_unique_tokens(tr_data)
        index_dict = self.get_indexing(token_dict)
        tr_refined_data = self.get_refined_data(max_len, index_dict, token_dict, tr_data)
        return tr_refined_data, token_dict, index_dict
    
    def get_data_for_validation(self, file_name, max_len, index_dict, token_dict):
        dp_obj = data_preparation()
        val_data_objs = dp_obj.data_load(file_name)
        val_data = dp_obj.tag_all_cues(val_data_objs)        
        val_refined_data = self.get_refined_data(max_len, index_dict, token_dict, val_data)

        return val_refined_data, val_data_objs
    
    def prepare_training_data(self, data, features_dict, index_dict, embed_dim=300):
        x = []  
        num_tokens = {}
        embed_dims = {}
        if "words" in features_dict: 
            x.append(data["sentences"]) 
            num_tokens["words"] = len(index_dict["word2index"])
            embed_dims["words"] = embed_dim        
        if "pos" in features_dict: 
            x.append(data["pos"])
            num_tokens["pos"] = len(index_dict["pos2index"])
            embed_dims["pos"] = embed_dim
        if "upos" in features_dict: 
            x.append(data["upos"])
            num_tokens["upos"] = len(index_dict["upos2index"])
            embed_dims["upos"] = embed_dim
        if "syntax" in features_dict: 
            x.append(data["syntax"])
            num_tokens["syntax"] = len(index_dict["syntax2index"])
            embed_dims["syntax"] = embed_dim
        if "sdep" in features_dict: 
            x.append(data["sdep"])
            num_tokens["sdep"] = len(index_dict["sdep2index"])
            embed_dims["sdep"] = embed_dim
        return x, num_tokens, embed_dims

        

class data_for_training():
    # Returns unique tokens from a list of list data 
    def unique_tokens(self, data_list_of_list, isLower = False):
        token_all = set()
        for sentence in data_list_of_list:
            for token in sentence:
                if isLower == False:
                    token_all.add(token)
                else:
                    token_all.add(token.lower())
        
        return list(token_all)   
    
    # Returns dictionary of data of unique tokens of features or labels
    def get_unique_tokens(self, data, isLower):
        token_dict = {}         
        
        if "sentences" in  data: token_dict["words"] = self.unique_tokens(data["sentences"], isLower)   # For all words in the data              
        if "lemma" in  data: token_dict["lemma"] = self.unique_tokens(data["lemma"], isLower) # For all lemmas in the data        
        if "pos" in  data: token_dict["pos"] = self.unique_tokens(data["pos"]) # For all PoSs in the data
        if "upos" in  data: token_dict["upos"] = self.unique_tokens(data["upos"]) # For all Universal PoSs in the data
        if "syntax" in  data: token_dict["syntax"] = self.unique_tokens(data["syntax"]) # For all syntax in the data
        if "sdep" in  data: token_dict["sdep"] = self.unique_tokens(data["sdep"])      # For all syntactic dependency in data
        if "cues" in  data: token_dict["cues"] = self.unique_tokens(data["cues"]) # For all Negation Cues in the data 
        if "cues_spec" in  data: token_dict["cues_spec"] = self.unique_tokens(data["cues_spec"])  # For all specific Negation Cues in the data
        if "scopes" in  data: token_dict["scopes"] = self.unique_tokens(data["scopes"]) # For all scopes in the data
        if "events" in  data: token_dict["events"] = self.unique_tokens(data["events"])
        
        
        return token_dict
    
    
    # Returns two dictionaries having received input a list of unique tokens. Two dictionaries are from token to index and index 2 token.
    def token_indexing(self, token_list, isPad =True, isUnknown=True):
        count = 0
        if isPad == True: count += 1
        if isUnknown == True: count += 1        
        
        token2index = {w:i+count for i, w in enumerate(token_list)} 
        if isPad == True: token2index[PAD] = 0
        if isUnknown == True: token2index[UNKNOWN] = 1 if isPad == True else 0
        
        index2token = {i:w for w, i in token2index.items()}        
        return token2index, index2token
        
    
    # Returns a dictionary that stores dictionaries of featues/labels with indicies.
    def get_indexing(self, token_dict):
        index_dict = {}
        if "words" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["words"], isPad, isUnknown)
            index_dict["word2index"] = token2index; index_dict["index2word"] = index2token            
            
        if "lemma" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["lemma"], isPad, isUnknown)
            index_dict["lemma2index"] = token2index; index_dict["index2lemma"] = index2token
            
        if "pos" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["pos"], isPad, isUnknown)
            index_dict["pos2index"] = token2index; index_dict["index2pos"] = index2token
            
        if "upos" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["upos"], isPad, isUnknown)
            index_dict["upos2index"] = token2index; index_dict["index2upos"] = index2token                    
        
        if "syntax" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["syntax"], isPad, isUnknown)
            index_dict["syntax2index"] = token2index; index_dict["index2syntax"] = index2token
            
        if "sdep" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["sdep"], isPad, isUnknown)
            index_dict["sdep2index"] = token2index; index_dict["index2sdep"] = index2token
            
        if "cues" in token_dict: 
            isPad =True; isUnknown=False
            token2index, index2token = self.token_indexing(token_dict["cues"], isPad, isUnknown)
            index_dict["cue2index"] = token2index; index_dict["index2cue"] = index2token
            
        if "cues_spec" in token_dict: 
            isPad =True; isUnknown=False
            token2index, index2token = self.token_indexing(token_dict["cues_spec"], isPad, isUnknown)
            index_dict["cueSpec2index"] = token2index; index_dict["index2cueSpec"] = index2token
            
        if "scopes" in token_dict: 
            isPad =True; isUnknown=False
            token2index, index2token = self.token_indexing(token_dict["scopes"], isPad, isUnknown)
            index_dict["scope2index"] = token2index; index_dict["index2scope"] = index2token
        
        if "events" in token_dict: 
            isPad =True; isUnknown=False
            token2index, index2token = self.token_indexing(token_dict["events"], isPad, isUnknown)
            index_dict["event2index"] = token2index; index_dict["index2event"] = index2token
        
        return index_dict
    
    
    # Returns the list of list of sentences with indicies. Each token of a sentence is replaced with it's 
    # corresponding index number.
    def get_sent_with_padding(self, sentence_list, unique_token_list, token2index, max_len, isLower = False):
        indexed_sent_list = []
        for sentence in sentence_list:
            token_list = []
            for token in sentence:
                if isLower == True: token = token.lower()
                if token in unique_token_list:
                    token_list.append(token2index[token] )
                else:
                    token_list.append(token2index[UNKNOWN])  #for unknown word
            indexed_sent_list.append(token_list)
            token_list = []
        
        indexed_sent_list = pad_sequences(maxlen=max_len, sequences=indexed_sent_list, padding="post", value=token2index[PAD])
        return indexed_sent_list
    
    
    # Returns indexed and padded data for all features and labels.
    def get_processed_data(self, max_len, index_dict, token_dict, data, isLower = False):
        processed_data = {}
        
        if "sentences" in  data: 
            processed_data["words"] = self.get_sent_with_padding(data["sentences"], token_dict["words"], index_dict["word2index"], max_len, isLower)              
        if "lemma" in  data: 
            processed_data["lemma"] = self.get_sent_with_padding(data["lemma"], token_dict["lemma"], index_dict["lemma2index"], max_len, isLower)
        if "pos" in  data: 
            processed_data["pos"] = self.get_sent_with_padding(data["pos"], token_dict["pos"], index_dict["pos2index"], max_len)
        if "upos" in  data: 
            processed_data["upos"] = self.get_sent_with_padding(data["upos"], token_dict["upos"], index_dict["upos2index"], max_len)
        if "syntax" in  data: 
            processed_data["syntax"] = self.get_sent_with_padding(data["syntax"], token_dict["syntax"], index_dict["syntax2index"], max_len)
        if "sdep" in  data: 
            processed_data["sdep"] = self.get_sent_with_padding(data["sdep"], token_dict["sdep"], index_dict["sdep2index"], max_len)            
        if "cues" in  data: 
            processed_data["cues"] = self.get_sent_with_padding(data["cues"], token_dict["cues"], index_dict["cue2index"], max_len)
        if "cues_spec" in  data: 
            processed_data["cues_spec"] = self.get_sent_with_padding(data["cues_spec"], token_dict["cues_spec"], index_dict["cueSpec2index"], max_len)
        if "scopes" in  data: 
            processed_data["scopes"] = self.get_sent_with_padding(data["scopes"], token_dict["scopes"], index_dict["scope2index"], max_len)
        if "events" in  data: 
            processed_data["events"] = self.get_sent_with_padding(data["events"], token_dict["events"], index_dict["event2index"], max_len)

        return processed_data
    
    
    def get_data_for_training(self, file_name, max_len, isLower, isIncludeNonCue, label_name):    
        # Generate Training data object
        dp_obj = data_preparation()
        tr_prep_obj = dp_obj.data_load(file_name)
        
        # Extract data for scope resolution
        detail_data_dict = dp_obj.get_data_details(tr_prep_obj)
        data_for_scope_event = dp_obj.data_for_scope_resolution(detail_data_dict, isIncludeNonCue)
        
        # Extract unique tokens of the features/labels
        token_dict = self.get_unique_tokens(data_for_scope_event, isLower)
        
        # Indexing the tokens of the features/labels
        index_dict = self.get_indexing(token_dict)    
        
        # Generate final Datasets
        proc_data = self.get_processed_data(max_len, index_dict, token_dict, data_for_scope_event, isLower)
        
        # Generate labels        
        num_labels = len(token_dict[label_name])
        labels = self.get_labels(proc_data[label_name], num_labels)
         
        return proc_data, labels, token_dict, index_dict
    
    
    def get_data_for_validation(self, file_name, max_len, index_dict, token_dict, isLower, isIncludeNonCue, label_name):
        dp_obj = data_preparation()
        tr_prep_obj = dp_obj.data_load(file_name)
        # Extract data for scope resolution
        detail_data_dict = dp_obj.get_data_details(tr_prep_obj)
        data_for_scope_event = dp_obj.data_for_scope_resolution(detail_data_dict, isIncludeNonCue)
        
        # Generate final Datasets
        proc_data = self.get_processed_data(max_len, index_dict, token_dict, data_for_scope_event, isLower)
        
        # Generate labels        
        num_labels = len(token_dict[label_name])
        labels = self.get_labels(proc_data[label_name], num_labels)    
        
        return proc_data, labels
    
    
    # Generates 2D to 3D array after one-hot-encoding on the labels data. 1st dimension: number of 
    # examples, 2nd dimension: label types (index of that type), 3rd dimension: one-hot-vector each type.
    def get_labels(self, labels, unique_labels):
        labels = [to_categorical(l, num_classes = unique_labels + 1) for l in  labels] # num_labels + 1, 1 is added for padding
        return np.array(labels)
    
    
    # Split data into training and test sets
    def data_split_basic(self, X,Y, test_parcent = 0.25 ):
        train_x, test_x, train_y, test_y = train_test_split(X, Y, shuffle=True, test_size=test_parcent)
        return train_x, test_x, train_y, test_y
    
    
    def prepare_training_data(self, data, features_dict, index_dict, embed_dim=300):
        x = []  
        num_tokens = {}
        embed_dims = {}
        if "words" in features_dict: 
            x.append(data["words"]) 
            num_tokens["words"] = len(index_dict["word2index"])
            embed_dims["words"] = embed_dim
        if "cues" in features_dict: 
            x.append(data["cues"])
            num_tokens["cues"] = len(index_dict["cue2index"])
            embed_dims["cues"] = embed_dim
        if "cues_spec" in features_dict: 
            x.append(data["cues_spec"])
            num_tokens["cues_spec"] = len(index_dict["cueSpec2index"])
            embed_dims["cues_spec"] = embed_dim
        if "scopes" in features_dict: 
            x.append(data["scopes"])
            num_tokens["scopes"] = len(index_dict["scope2index"])
            embed_dims["scopes"] = embed_dim
        if "events" in features_dict: 
            x.append(data["events"])
            num_tokens["events"] = len(index_dict["event2index"])
            embed_dims["events"] = embed_dim
        if "pos" in features_dict: 
            x.append(data["pos"])
            num_tokens["pos"] = len(index_dict["pos2index"])
            embed_dims["pos"] = embed_dim
        if "upos" in features_dict: 
            x.append(data["upos"])
            num_tokens["upos"] = len(index_dict["upos2index"])
            embed_dims["upos"] = embed_dim
        if "syntax" in features_dict: 
            x.append(data["syntax"])
            num_tokens["syntax"] = len(index_dict["syntax2index"])
            embed_dims["syntax"] = embed_dim
        if "sdep" in features_dict: 
            x.append(data["sdep"])
            num_tokens["sdep"] = len(index_dict["sdep2index"])
            embed_dims["sdep"] = embed_dim
        return x, num_tokens, embed_dims
    
    def prepare_training_data_elmo(self, data, features_dict, index_dict, embed_dim = 300):
        x = []  
        num_tokens = {}
        embed_dims = {}
        
        if "words" in features_dict:             
            data["words"] = [[index_dict["index2word"][i] for i in sent] for sent in data["words"] ]  # making like [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']], this format is required for the elmo signature signature="tokens"
            #data["words"] = [" ".join([index_dict["index2word"][i] for i in sent]) for sent in data["words"] ]   #making like ['this is a test','this is another test'], this format is required for the elmo signature signature="default"            
            
            x.append(data["words"]) 
            num_tokens["words"] = len(index_dict["word2index"])
            embed_dims["words"] = embed_dim
        if "cues" in features_dict:             
            x.append(data["cues"])
            num_tokens["cues"] = len(index_dict["cue2index"])
            embed_dims["cues"] = embed_dim
        if "cues_spec" in features_dict:             
            x.append(data["cues_spec"])
            num_tokens["cues_spec"] = len(index_dict["cueSpec2index"])
            embed_dims["cues_spec"] = embed_dim
        if "scopes" in features_dict:             
            x.append(data["scopes"])
            num_tokens["scopes"] = len(index_dict["scope2index"])
            embed_dims["scopes"] = embed_dim
        if "events" in features_dict:             
            x.append(data["events"])
            num_tokens["events"] = len(index_dict["event2index"])
            embed_dims["events"] = embed_dim
        if "pos" in features_dict:             
            x.append(data["pos"])
            num_tokens["pos"] = len(index_dict["pos2index"])
            embed_dims["pos"] = embed_dim
        if "upos" in features_dict:             
            x.append(data["upos"])
            num_tokens["upos"] = len(index_dict["upos2index"])
            embed_dims["upos"] = embed_dim
        if "syntax" in features_dict:             
            x.append(data["syntax"])
            num_tokens["syntax"] = len(index_dict["syntax2index"])
            embed_dims["syntax"] = embed_dim
        if "sdep" in features_dict:             
            x.append(data["sdep"])
            num_tokens["sdep"] = len(index_dict["sdep2index"])
            embed_dims["sdep"] = embed_dim
        return x, num_tokens, embed_dims

