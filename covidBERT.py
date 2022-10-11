import pandas as pd

data = pd.read_csv("covid19_tweeter_dataset.csv")

data['length'] = data['Tokanize_tweet'].apply(
    lambda row: min(len(row.split(",")), len(row)) if isinstance(row, str) else None
)

import torch
import numpy as np 
import pandas as pd 
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
from scipy.spatial.distance import cosine 
import tokenizers 
import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA


import os 
import re
import json
import pickle 


model = AutoModelForMaskedLM.from_pretrained("Shushant/nepaliBERT", output_hidden_states = True, return_dict = True, output_attentions = True)

tokenizers = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")

def get_bert_embedding_sentence(input_sentence):
    md = model
    tokenizer = tokenizers
#     md = local_model
#     tokenizer = local_tokenizers
    marked_text = " [CLS] " + input_sentence + " [SEP] "
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens) 
    
    # Convert inputs to Pytorch tensors
    tokens_tensors = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        outputs = md(tokens_tensors, segments_tensors)
        # removing the first hidden state
        # the first state is the input state 

        hidden_states = outputs.hidden_states
#         print(hidden_states[-2])
        # second_hidden_states = outputs[2]
    # hidden_states has shape [13 x 1 x 22 x 768]

    # token_vecs is a tensor with shape [22 x 768]
#     token_vecs = hidden_states[-2][0]
    # get last four layers
#     last_four_layers = [hidden_states[i] for i in (-1,-2, -3,-4)]
    # cast layers to a tuple and concatenate over the last dimension
#     cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
#     print(cat_hidden_states.shape)
    token_vecs = hidden_states[-2][0]
    # take the mean of the concatenated vector over the token dimension
#     sentence_embedding = torch.mean(cat_hidden_states, dim=0).squeeze()

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
#     sentence_embedding = torch.mean(token_vecs, dim=1)
    return sentence_embedding.numpy()


data['word_embeddings'] = data['Tweet'].apply(get_bert_embedding_sentence)

data.to_csv("embeddings_data.csv")

X,y = data['word_embeddings'], data['Label']


train_X, test_X, train_y, test_y = train_test_split(X,y, test_size = 0.2, random_state = 420)



svc = SVC(probability=True)


svc.fit(train_X.tolist(), train_y)
#svc.fit(train_X, train_y)


# In[22]:


svc_pred = svc.predict(test_X.tolist())
# svc_pred = svc.predict(test_X)


# In[23]:


print(confusion_matrix(test_y, svc_pred))


# In[24]:


print(classification_report(test_y, svc_pred))



# In[ ]:



from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(train_X.tolist(), train_y)
gbc.score(test_X.tolist(), test_y)


# In[26]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(train_X.tolist(), train_y)



clf.score(test_X.tolist(), test_y)


# In[27]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train_X.tolist(), train_y)
gnb.score(test_X.tolist(), test_y)


# In[28]:


gnb_pred = gnb.predict(test_X.tolist())
print(confusion_matrix(test_y, gnb_pred))

from skrvm import RVC

clf = RVC()

clf.fit(train_X.tolist(), train_y)

clf.score(test_X.tolist(), test_y)

import pickle
# save the model to disk
filename = 'rvc_model.sav'
pickle.dump(clf, open(filename, 'wb'))

