
# coding: utf-8

# In this program, we assemble a body of text extracted from semantically related articles in Wikipedia, and then run it through a POS tagger. We then proceed to use these POS tags as labels in a supervised sequence labeling task using a Long Short Term Memory Network (LSTM).  

# In[1]:

# Dedicated to the spirit of Mr Mojo Risin 
# Omid Rohanian

import wikipedia, string, re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from nltk.tag import UnigramTagger
from nltk.corpus import stopwords
import sklearn.preprocessing
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[2]:

doors = ['The Doors', 'Jim Morrison', 'Ray Manzarek', 'Robby Krieger', 'John Densmore', 'Strange Days (album)',
         'The Doors (album)', 'Absolutely Live (The Doors album)', 'L.A. Woman', 'Light My Fire', 'Hello, I Love You',
         'Touch Me (The Doors song)', 'Waiting for the Sun', 'The Soft Parade', 'Morrison Hotel', 
         'Other Voices (The Doors album)', 'Full Circle (The Doors album)', 'An American Prayer', 'The Doors – 30 Years Commemorative Edition',
         'Alabama Song', 'The End (The Doors song)', 'Counterculture of the 1960s', 'Break On Through (To the Other Side)',
         'Live at the Matrix 1967', 'People Are Strange', 'Roadhouse Blues', 'Riders on the Storm', 'Love Me Two Times']
text = ''
for door in doors:
    text += wikipedia.page(door).content

sents = text.split('\n')
    
# Each sentence is a list of words 
def preprocess(sents):
    sents = [sent.translate(str.maketrans('','', string.punctuation)).strip(string.digits).lower() for sent in sents]
    sents = [word_tokenize(sent) for sent in sents]
    return [[word for word in sent if word not in set(stopwords.words('english'))] for sent in sents]

# We will tag all the words with unique POS tags and later use these as labels for our classification task

tagger = UnigramTagger(brown.tagged_sents(categories='news'))
sents = preprocess(sents)
words = list(set([word for sent in sents for word in sent]))
pos_tags = dict(tagger.tag(words))
maxlengths=max([len(s) for s in sents])


# In[3]:

words_dic={word:i+1 for (i,word) in enumerate(words)}
words_tags_pairs=[tagger.tag(s) for s in sents]

y=[[w[1] for w in wp] for wp in words_tags_pairs]
X=[[words_dic[w[0]] for w in wp] for wp in words_tags_pairs]
X=[[x+[0]*(maxlengths-len(x))] for x in X]

all_tags=set([x for s in y for x in s])
all_tags_dic={t:i for (t,i) in zip (all_tags,range(1,len(all_tags)+1))}
all_tags_dic["eos"]=0

#One-hot encode the labels 
y_num=[[all_tags_dic[t] for t in s]+[0]*(maxlengths-len(s)) for s in y]
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(36))
y_onehot=[label_binarizer.transform(s) for s in y_num]


# In[4]:

# preparing the train and test data to be fed into the network 
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=seed)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train=np.reshape(X_train,[len(X_train),maxlengths])
X_test=np.reshape(X_test,[len(X_test),maxlengths])


# In[5]:

# Building the LSTM network 
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import TimeDistributed,Input,Embedding,Masking
from keras.layers.recurrent import LSTM, GRU

inputs=Input(shape=[maxlengths,])
masked_inputs=Masking(mask_value=0)(inputs)
embedding_words=Embedding(input_dim=len(words_dic)+1,output_dim=128)(masked_inputs)


lstm=LSTM(units=50,return_sequences=True)(embedding_words)
dropout=Dropout(0.5)(lstm)


outputs=TimeDistributed(Dense(36,activation="softmax"))(dropout)
print(outputs)
model=Model(inputs=inputs,outputs=outputs)

#compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# fit the model on the training data
model.fit(X_train,y_train,
          batch_size=len(X_train),epochs=30,
          validation_split=0.2)
#evaluate the model on the test data (for faster training we have used the size of the whole data as batch size)
model.evaluate(X_test,y_test,batch_size=len(X_test))


# In[ ]:



