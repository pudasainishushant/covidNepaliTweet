#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("covid19_tweeter_dataset.csv")


# In[3]:


data


# In[4]:


data['Tweet'][30]


# In[5]:


data['length'] = data['Tokanize_tweet'].apply(
    lambda row: min(len(row.split(",")), len(row)) if isinstance(row, str) else None
)


# In[6]:


max(data['length'].tolist())


# In[7]:


data["Tweet"][1]


# In[8]:


data.head()


# In[9]:


data["Tokanize_tweet"][1]


# In[10]:


data["Label"].value_counts()


# In[11]:


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
from sklearn.preprocessing import StandardScaler
# import snowballstemmer
import numpy
import os 
import re
import json
import pickle 


# In[12]:


model = AutoModelForMaskedLM.from_pretrained("Shushant/nepaliBERT", output_hidden_states = True, return_dict = True, output_attentions = True)


# In[13]:


tokenizers = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")


# pickle.dump(model, open('bert_embeddings','wb'))

# pickle.dump(tokenizers, open('bert_tokenizers','wb'))

# model = pickle.load(open('bert_embeddings','rb'))
# tokenizers= pickle.load(open('bert_tokenizers','rb'))

# In[14]:


tokenizers.tokenize("के मौजुदा लोकतान्त्रिक व्यवस्था राज्य पुनःसंरचनासँग जोडिएका हिजोका सवालहरूलाई यथास्थितिमा छोडेर सबल होला?")


# In[15]:


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


# In[16]:


data["Tweet"][0]


# In[17]:


data['word_embeddings'] = data['Tweet'].apply(get_bert_embedding_sentence)


# In[26]:


data.to_csv("embeddings_data.csv")


# In[18]:


X,y = data['word_embeddings'], data['Label']


# In[19]:


train_X, test_X, train_y, test_y = train_test_split(X,y, test_size = 0.2, random_state = 420)


# train_df,test_df = train_test_split(data, test_size = 0.2, random_state = 420)

# train_df.to_csv("train.csv")

# test_df.to_csv("test.csv")

# In[20]:


svc = SVC(probability=True)


# In[21]:


svc.fit(train_X.tolist(), train_y)
#svc.fit(train_X, train_y)


# In[22]:


svc_pred = svc.predict(test_X.tolist())
# svc_pred = svc.predict(test_X)


# In[23]:


print(confusion_matrix(test_y, svc_pred))


# In[24]:


print(classification_report(test_y, svc_pred))


# In[25]:


accuracy_score(test_y, svc_pred)


# In[ ]:


sent = "नराम्रो"
predicted_label = svc.predict_proba(np.array(get_bert_embedding_sentence(sent).tolist()).reshape(1,-1))[0]
if predicted_label[0]<predicted_label[1]:
    print(f'{sent} is negative sentiment')
else:
    print(f'{sent} is positive sentiment')


# In[25]:


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


# ## Deep Learning

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


data['word_embeddings'][0]


# In[29]:


from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[51]:


Y_array = data['Label'].to_list()


# In[52]:


Y = np.array(Y_array)


# In[53]:


Y


# In[65]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[41]:


x_array = data['word_embeddings'].to_list()


# In[46]:


x = np.array(x_array)


# In[58]:


x.shape


# In[76]:


dummy_y.shape


# In[77]:


x.shape


# In[86]:


train_x, test_x, train_Y, test_Y = train_test_split(x,dummy_y, test_size = 0.2, random_state = 420)


# In[115]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=768, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25,activation="relu"))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
# estimator = KerasClassifier(build_fn=baseline_model, epochs=2, batch_size=5, verbose=1)
# kfold = KFold(n_splits=2, shuffle=True)
# results = cross_val_score(estimator, x,dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[116]:


model = baseline_model()
model.fit( train_x,train_Y, epochs=20, batch_size=10)


# In[ ]:


x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)


# In[117]:


model.evaluate(test_x,test_Y)


# In[118]:


predictions = model.predict(test_x)


# In[119]:


pred = np.argmax(predictions, axis=1)


# In[120]:


pred


# In[121]:


encoded_test_y = np.argmax(test_Y,axis=1)


# In[122]:


print(confusion_matrix(encoded_test_y, pred))


# In[123]:


print(classification_report(encoded_test_y, pred))


# In[48]:


from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X = train_X.to_list()
y = train_y

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


# In[49]:


clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


# In[50]:


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean() 


# In[ ]:





# ## Finetuning with NepBERT

# In[16]:


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("Shushant/nepaliBERT", num_labels=3)


# In[17]:


import numpy as np
import evaluate

metric = evaluate.load("accuracy")


# In[18]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[19]:


from datasets import load_dataset

dataset = load_dataset("Shushant/CovidNepaliTweets")
dataset["train"][100]


# In[28]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")

max_length = 61
def tokenize_function(examples):
    return tokenizer(examples['Text'], padding='max_length', truncation=True, max_length=max_length)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# In[29]:


tokenized_datasets['train'][0]


# In[30]:


len(tokenized_datasets['train'][0]['attention_mask'])


# In[35]:


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# In[36]:


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")


# In[37]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)


# In[38]:


trainer.train()


# In[ ]:




