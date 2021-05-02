import numpy as np
import nltk
import scipy
import csv
import os
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from collections import Counter
from nltk.stem import PorterStemmer

def read(path,filename):
    data=[]
    read = os.listdir(path)
    for fn in read:
        if fn == filename:
            with open(fn, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) !=0:
                        data.append(row)
                    
    return data

class Featurizer():
           
    @staticmethod
    def number_of_words(text):
        w=word_tokenize(text)
        l=len(w)
        yield l
            
    @staticmethod
    def frequent_words(text):
        freq = [".","?",",","''","people","would","like","one","dont","time","state"]
        wt=word_tokenize(text)
        stemmer=PorterStemmer()
        stopwords = set(nltk.corpus.stopwords.words('english'))
        new_t = [stemmer.stem(word) for word in wt if word not in stopwords]
        for word in freq:
            if word in new_t:
                yield 1
        
    @staticmethod
    def POS_tag_noun(text):
        tags = nltk.pos_tag(word_tokenize(text))
        noun = sum(1 for w, tag in tags if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNP')
        yield ('NOUN', noun)
            
    @staticmethod    
    def POS_tag_verb(text):
        tags = nltk.pos_tag(word_tokenize(text))
        verb = sum(1 for w, tag in tags if tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN')
        yield ('VERB',verb)
            
    @staticmethod
    def POS_tag_adj(text):
        tags = nltk.pos_tag(word_tokenize(text))
        adj = sum(1 for w, tag in tags if tag == 'JJ' or tag == 'JJR' or tag == 'JJS')
        yield ('ADJ',adj)
            
    @staticmethod
    def POS_tag_adv(text):
        tags = nltk.pos_tag(word_tokenize(text))
        adv = sum(1 for w, tag in tags if tag == 'RB' or tag == 'RBR' or tag == 'RBS')
        yield ('ADV',adv)

    @staticmethod      
    def unigrams(text):
        
        wt = word_tokenize(text)
        stemmer=PorterStemmer()
        stopwords = set(nltk.corpus.stopwords.words('english'))
        new_t = [stemmer.stem(word) for word in wt if word not in stopwords]
        unigrams=ngrams(new_t,1) 
        for u in unigrams:
            yield u
        """
        unigrams=ngrams(word_tokenize(text),1)
        for u in unigrams:
            yield u
        """ 

    @staticmethod      
    def bigrams(text):
        
        wt = word_tokenize(text)
        stopwords = set(nltk.corpus.stopwords.words('english'))
        new_t = [word for word in wt if word not in stopwords]
        bigrams=ngrams(new_t,2) 
        for b in bigrams:
            yield b
        """
        bigrams=ngrams(word_tokenize(text),2)
        for b in bigrams:
            yield b
        """   
    @staticmethod      
    def trigrams(text):
        
        wt = word_tokenize(text)
        stopwords = set(nltk.corpus.stopwords.words('english'))
        new_t = [word for word in wt if word not in stopwords]
        trigrams=ngrams(new_t,3)
        for t in trigrams:
            yield t
        """
        trigrams=ngrams(word_tokenize(text),3)
        for t in trigrams:
            yield t
        """
        
    @staticmethod      
    def missing_words(text):
        missing_list = ["nigger","yucki","hemorrhoid","retard","garbageman",
                        "ars","screwi","motherfuck","gun","corrupt","nut","charge",
                        'black','shoot','homosexu','crazi','pathet','stupid',
                        'hypocrit','ignor','transgend','jerk','bigot','kill',
                        'incompet','darn','thug','idiot','foolish','ball','gay',
                        'damn','moron','lesbian','clown','rape','silli',
                        'rotten','rot','fool','ridicul','parasit','suck',
                        'miser','punch','asshol','troll','disgust','shit',
                        'coward','dirti','peni','loser','ass','arrog','butt',
                        'liar','scum','pussi','crap','garbag','prostitut',
                        'oppress','freak','douch','idioci','dumb','disgrac',
                        'sucker','scumbag','bum','fuck','friggin','dumbest',
                        'bullshit','bastard','dumber','imbecil','bitch','vagina',
                        'hate',"white","trigger","blame","trash","drug","predator",
                        "arm","fire","sick","dick","!","dear","thank"]
        
        wt = word_tokenize(text)
        stemmer=PorterStemmer()
        stopwords = set(nltk.corpus.stopwords.words('english'))
        new_t = [stemmer.stem(word) for word in wt if word not in stopwords]
        for word in missing_list:
            if word in new_t:
                yield 1
                
    @staticmethod      
    def points(text):
        wt = word_tokenize(text)
        stemmer=PorterStemmer()
        stopwords = set(nltk.corpus.stopwords.words('english'))
        new_t = [stemmer.stem(word) for word in wt if word not in stopwords]
        for word in new_t:
            if '.' in word:
                yield 1
   
    #feature_functions = ['bag_of_words','number_of_words','POS_tag_noun','POS_tag_verb','POS_tag_adj','POS_tag_adv','bigrams','trigrams']
    #feature_functions = ['bag_of_words']
    #feature_functions = ['number_of_words']
    #feature_functions = ['bigrams']
    feature_functions = ['unigrams','missing_words']
    #feature_functions = ['trigrams']
    #feature_functions = ["frequent_words"]
    #feature_functions=['POS_tag_noun','POS_tag_verb','POS_tag_adj','POS_tag_adv']
    #feature_functions = ['bigrams','trigrams']
    #feature_functions = ['bigrams','trigrams','POS_tag_noun','POS_tag_verb','POS_tag_adj','POS_tag_adv']
    
    def __init__(self):
        self.labels = {}
        self.labels_by_id = {}
        self.features = {}
        self.next_feature_id = 0
        self.next_label_id = 0
        self.c_list = []
        
    def list_values(self,events,c):
        length=0
        for i in range(c+1):
            length += len(events[i])
        return length
    

    def to_sparse(self, events,values):
        """convert sets of ints to a scipy.sparse.csr_matrix"""
        data, row_ind, col_ind = [], [], []
        n = 0
        for event_index, event in enumerate(events):
            for feature in event:
                data.append(values[n])
                n += 1
                row_ind.append(event_index)
                col_ind.append(feature)
                
        n_features = self.next_feature_id
        n_events = len(events)
        matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(n_events, n_features)).toarray()
        return matrix

    def featurize(self,dataset, allow_new_features=False):
        frequency, f_with_id, freq_count = [], [], []
        events, values, labels, id , label_names, feature_names = [], [], [], [], [], ["id"]
        for c, [text_id,text, label] in tqdm(enumerate(dataset)):
            id.append(int(text_id))
            if label not in self.labels:
                if int(label) == 0:
                    self.next_label_id = 0
                    self.labels[label] = self.next_label_id
                    self.labels_by_id[self.next_label_id] = label
                
                if int(label) == 1:
                    self.next_label_id = 1
                    self.labels[label] = self.next_label_id
                    self.labels_by_id[self.next_label_id] = label
                
            labels.append(self.labels[label])
            events.append(set())
            
            for function_name in Featurizer.feature_functions:
                function = getattr(Featurizer, function_name)
                if  function_name == 'missing_words' :
                    for num in function(text):
                        f = 'missing_words'
                        if f not in self.features:
                            feature_names.append(f)
                            if not allow_new_features:
                                continue
                            self.features[f] = self.next_feature_id
                            self.next_feature_id += 1
            
                        feat_id = self.features[f]
                        events[-1].add(feat_id)
                        values.append(1)
                        
    
                
                if  function_name == 'number_of_words' :
                    for length in function(text):
                        f1= 'number_of_words'
                        if f1 not in self.features:
                            feature_names.append(f1)
                            if not allow_new_features:
                                continue
                            self.features[f1] = self.next_feature_id
                            self.next_feature_id += 1
                        
                        feat_id = self.features[f1]
                        events[-1].add(feat_id)
                        values.append(length)
            
                if  function_name == 'frequent_words' :
                    for num in function(text):
                        f2 = 'frequent_words'
                        if f2 not in self.features:
                            feature_names.append(f2)
                            if not allow_new_features:
                                continue
                            self.features[f2] = self.next_feature_id
                            self.next_feature_id += 1
            
                        feat_id = self.features[f2]
                        events[-1].add(feat_id)
                        values.append(1)
            
                if  function_name in ['POS_tag_noun','POS_tag_verb','POS_tag_adj','POS_tag_adv'] :
                    for (POS,s) in function(text):
                        if POS not in self.features:
                            feature_names.append(POS)
                            if not allow_new_features:
                                continue
                            self.features[POS] = self.next_feature_id
                            self.next_feature_id += 1
                        feat_id = self.features[POS]
                        events[-1].add(feat_id)
                        values.append(s)

                if function_name in ['unigrams','bigrams','trigrams']:
                    for feature in function(text):
                        frequency.append(feature)
                f_with_id.append([text_id,c,frequency])
                self.c_list += frequency
                frequency=[]
      
        counter=Counter(self.c_list)
        freq_count=[i for i in counter if counter[i] > 5]
        for function_name in Featurizer.feature_functions:
            function = getattr(Featurizer, function_name)            
            if function_name in ['unigrams','bigrams','trigrams']:
                for [i,c,f] in tqdm(f_with_id):
                    for feature in freq_count:
                        if feature not in self.features:
                            if allow_new_features == True:
                                feature_names.append(str(feature))
                            if allow_new_features == False:
                                feature_names.append(str(feature))  
                            if not allow_new_features:
                                continue
                            self.features[feature] = self.next_feature_id
                            self.next_feature_id += 1
                        feat_id = self.features[feature]
                        if feature in f:
                            n=self.list_values(events, c)
                            events[c].add(feat_id)
                            values.insert(n,1)
                            
        events_sparse = self.to_sparse(events,values)
        #print(events_sparse)
        labels_array = np.array(labels)
        #print(labels_array)
        id = np.array([id]).T
        matrix_with_id = np.append(id,events_sparse,axis=1)
        print("Saving")
        
        if allow_new_features == True:
            np.save("matrix_with_id_train",matrix_with_id)
            np.save("labels_array_train",labels_array)
            with open('feature_names_train.txt', 'w') as file:
                for name in feature_names:
                    file.write('%s\n' % name)
                           
            with open('label_names_train.txt', 'w') as file:
                file.write('\n'.join('{} {}'.format(x[0],x[1]) for x in label_names))
                           
        if allow_new_features == False:
            np.save("matrix_with_id_val",matrix_with_id)
            np.save("labels_array_val",labels_array)
            with open('label_names_val.txt', 'w') as file:
                file.write('\n'.join('{} {}'.format(x[0],x[1]) for x in label_names))
            
        return events_sparse.shape, labels_array.shape
        
path=os.getcwd()
featurizer = Featurizer()

filename="train_set_bin_50000.csv"
dataset=read(path,filename)
print("Featurizer train set")
train_features, train_labels = featurizer.featurize(dataset, allow_new_features=True)
print(train_features, train_labels)

filename2="test_set_bin_5000.csv"
dataset2=read(path,filename2)
print("Featurizer validation set")
val_features, val_labels = featurizer.featurize(dataset2, allow_new_features=False)
print(val_features, val_labels)

