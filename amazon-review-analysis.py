#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# IMPORTING ALL THE NECCESSARY MODULES

# In[3]:


import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from textblob import TextBlob


# Reading the datasets of the four companies Oneplus,Apple,Vivo and Samsung on which we will perform EDA to see the problems the companies are facing .

# In[5]:


Oneplus =pd.read_csv("Oneplus.csv")
Iphone=pd.read_csv("Iphone.csv")
Samsung=pd.read_csv("Samsung.csv")
Vivo=pd.read_csv('Vivo.csv')


# Adding Company column to all the DataFrames before we merge into main DataFrame.

# In[6]:


Iphone['Company']='Apple'
Iphone


# In[7]:


Oneplus['Company']='Oneplus'
Oneplus


# In[8]:


Samsung['Company']='Samsung'
Samsung


# In[9]:


Vivo['Company']='Vivo'
Vivo


# Now, we will create a main DataFrame called 'result' to merge all the datasets into it.

# In[10]:


#Create a empty DataFrame 
result=pd.DataFrame()
s=[Vivo,Iphone,Samsung,Oneplus]
#Append all the other companies review data into Result DataFrame.
for i in s:
    result=result.append(i);
result.reset_index(inplace=True)
result


# Using TextBlob we will Sentiment of the following reviews and based on that score we will classify sentiment into 'Positive', 'Negative' and 'Neutral'. Before using Textblob we will clean the review to get more accurate sentiment score.

# In[11]:


def get_sentiment_textblob(message):
    clean_message=' '.join(re.sub('\n', " ", message).split())
    analysis=TextBlob(clean_message)
    score=analysis.sentiment.polarity 
    if score > 0:
        return "Positive"
    else: 
        return "Negative"


# Now, we will create a seperate column of Sentiment which we will get from the above defined function when we process our review to get Setniment using TextBlob library.

# In[12]:


get_sentiment_textblob(result['Comment'][0])


# In[13]:


result['Sentiment_Textblob_Comment']=result['Comment'].apply(get_sentiment_textblob)
result['Sentiment_Textblob_Title']=result['Title'].apply(get_sentiment_textblob)


# In[14]:


result


# # **DATA PREPROCESSING**

# We will define a function for Preprocessing our review.Preprocessing pipeline includes:
# 1. Lower casing the text.
# 2. Tokenize into words.
# 3. Filtering out words by removing punctuations,'@','/n'etc.
# 4. Removing Stopwords.
# 5. Lemmatization.

# In[15]:


#def preprocess(message):
 #   message=message.lower()
  #  token=word_tokenize(message)
    
    #remove some negation words from stopwords list
   # rem_sw=["no","not"]
    #stop_words=set([word for word in stopwords.words('english') if word not in rem_sw])
    #clean_review=[word for word in text if word not in stop_words]
    #clean_review_text=[word for word in clean_review if len(word)>=2]
    #lemmatizer=WordNetLemmatizer()
    #final_review=[lemmatizer.lemmatize(word) for word in clean_review_text]
    #final=' '.join(final_review)
    #return final
import string
punc = set(string.punctuation)

def preprocess(text):
    # Convert the text into lowercase
    text = text.lower()
    # Split into list
    wordList=text.split()
    #print(wordList)
    #wordList = word_tokenize(text)
    # Remove punctuation
    wordList = ["".join(x for x in word if (x=="'")|(x not in punc)) for word in wordList]
    wordList=[t for t in wordList if re.match(r'^[a-z]',t)]
    # Remove stopwords
    wordList = [word for word in wordList if word not in stopwords.words('english')]
    wordList=[word for word in wordList if len(word)>=2]
    # Lemmatisation
    lemmatizer=WordNetLemmatizer()
    wordList = [lemmatizer.lemmatize(word) for word in wordList]
    return " ".join(wordList)


# In[16]:


result['Comment']=[preprocess(sent) for sent in result['Comment']]
result['Title']=[preprocess(sent) for sent in result['Title']]


# In[17]:


result


# In[18]:


#Coverting the Rating string into integer for predict Sentiment from Ratings.
result['Rating']=[int(t[0]) for t in result.Rating]


# In[19]:


result


# We will define a function to get Sentiment of user by the Ratings he gave for the product. The Sentiment Classification according to the ratings are as follows:
# 1. 'Negative' Sentiment for Ratings in range [1,2].
# 2. 'Neutral' Sentiment for Ratings equal to 3.
# 3. 'Positive' Sentiment for Ratings in range [4,5].

# In[20]:


def get_sentiment(x):
    if(x<=2):
        return "Negative"
    else:
        return "Positive"


# In[21]:


#Convert Rating string into integer.
result['Sentiment_Rating']=result['Rating'].apply(get_sentiment)


# In[22]:


result


# In[23]:


# Percentage of Sentiment using Textblob and Sentiment using Rating do not match
count=result[result['Sentiment_Textblob_Comment']!=result['Sentiment_Rating']]
print(count.shape,result.shape)
print((count.shape[0]/result.shape[0])*100)


# # Data Visualisation

# **Plotting Average rating per Brand**

# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
# Average rating per brand
ax = result.groupby("Company").mean()["Rating"].sort_values().plot(kind="barh",
                                                                figsize=(8,5), 
                                                                title="Average rating per Brand")
plt.show()


# **Visualising Count of Sentiment according to Ratings of Different Companies**

# In[25]:


plt.figure(figsize=(18,8))
sns.countplot(x = 'Company', hue = 'Sentiment_Rating', data = result)
plt.xlabel('Moods', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.title('Count of Moods', fontsize = 24)


# Segregating the result DataFrame into Comapny wise DataFrame.

# In[26]:


Op=result[result['Company']=='Oneplus']
Ip=result[result['Company']=='Apple']
Vivo=result[result['Company']=='Vivo']
Samsung=result[result['Company']=='Samsung']


# Joining all Positive and Negative reviews for counting Bigrams frequencies. 

# In[27]:


Op_pos=' '.join(Op[(Op['Sentiment_Rating']=='Positive')]['Comment'])
Ip_pos=' '.join(Ip[(Ip['Sentiment_Rating']=='Positive')]['Comment'])
Ip_neg=' '.join(Ip[(Ip['Sentiment_Rating']=='Negative')]['Comment'])
Op_neg=' '.join(Op[(Op['Sentiment_Rating']=='Negative')]['Comment'])
Vivo_pos=' '.join(Vivo[(Vivo['Sentiment_Rating']=='Positive')]['Comment'])
Sam_pos=' '.join(Samsung[(Samsung['Sentiment_Rating']=='Positive')]['Comment'])
Vivo_neg=' '.join(Vivo[(Vivo['Sentiment_Rating']=='Negative')]['Comment'])
Sam_neg=' '.join(Samsung[(Samsung['Sentiment_Rating']=='Negative')]['Comment'])
result_pos=' '.join(result[result['Sentiment_Rating']=='Positive']['Comment'])
result_neg=' '.join(result[result['Sentiment_Rating']=='Negative']['Comment'])


# **WORDCLOUD of Bigrams using Bigram_Frequency**

# In[29]:


#  Create Bigram_frequency Dictionary.
from wordcloud import WordCloud, ImageColorGenerator
def word_freq_dict(text):
    # Convert text into word list
    stopwords=['amazon','oneplus','iphone','apple','vivo','samsung','plus','nord','phone','mobile','good']
    wordList=text.split()
    wordList=[word for word in wordList if word not in stopwords]
    bigram=nltk.bigrams(wordList)
    fdist = nltk.FreqDist(bigram)
    # Generate bigram freq dictionary
    wordFreqDict={k[0]+' '+k[1]:v for k,v in fdist.items()}
    return wordFreqDict


# In[30]:


def wordcloud_from_frequency(word_freq_dict, title, figure_size=(10, 6)):
    wordcloud.generate_from_frequencies(word_freq_dict)
    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title)
    plt.show()


# In[31]:


# Define a function to plot top15 positive words and top15 negative words in a grouped bar plot (from dictionaries)
def topn_wordfreq_bar_both(pos_word_freq_dict, neg_word_freq_dict, pos_num_doc, neg_num_doc, topn, title1, title2,palette1,palette2,Org,height=10):
    # Transform positive word frequency into DF
    df_pos = pd.DataFrame.from_dict(pos_word_freq_dict, orient="index").sort_values(by=0, ascending=False).head(topn)
    df_pos.columns = ["frequency"]
    df_pos["frequency"] = df_pos["frequency"] / pos_num_doc
    df_pos["label"] = "Positive"
    df_pos['Company']=Org
    df_pos.reset_index(inplace=True)
    # Transform negative word frequency into DF
    df_neg = pd.DataFrame.from_dict(neg_word_freq_dict, orient="index").sort_values(by=0, ascending=False).head(topn)
    df_neg.columns = ["frequency"]
    df_neg["frequency"] = df_neg["frequency"] / neg_num_doc
    df_neg["label"] = "Negative"
    df_neg['Company']=Org
    df_neg.reset_index(inplace=True)
    # Plot
    print(df_pos)
    sns.catplot(x="index", y="frequency", hue="label", data=df_pos, 
                kind="bar",
                palette=palette1,
                height=height,aspect=2,
                legend_out=True)
    plt.title(title1+Org)
    plt.show()
    print(df_neg)
    sns.catplot(x="index", y="frequency", hue="label", data=df_neg, 
                kind="bar",
                palette=palette2,
                height=height,aspect=2,
                legend_out=True)
    plt.title(title2+Org)
    plt.show()
    return df_pos,df_neg


# In[32]:


# Plot wordclouds for latest 1000 reviews for Apple
Ip_pos_word_freq = word_freq_dict(Ip_pos)
wordcloud = WordCloud(width=5000, 
                      height=3000, 
                      max_words=200, 
                      colormap="Blues",
                      background_color="white")
wordcloud_from_frequency(Ip_pos_word_freq, "Most Frequent Words in the Latest 1000 Positive Reviews for Apple")


# In[33]:


# Plot wordclouds for latest 1000 negativereviews for Apple
Ip_neg_word_freq = word_freq_dict(Ip_neg)
wordcloud = WordCloud(width=5000, 
                      height=3000, 
                      max_words=200, 
                      colormap="Blues",
                      background_color="Black")
wordcloud_from_frequency(Ip_neg_word_freq, "Most Frequent Words in the Latest 1000 Negative Reviews for Apple")


# In[34]:


#Plotting top 15 positive and negative words for Apple
Apple_top_pos,Apple_top_neg=topn_wordfreq_bar_both(Ip_pos_word_freq, Ip_neg_word_freq, 
                       min(sum(Ip['Sentiment_Rating']=='Positive'), 1000), 
                       min(sum(Ip['Sentiment_Rating']=='Negative'), 1000), 
                       15, 
                       "Top15 Frequent Words in Latest Positive for","Top15 Frequent Words in Latest Negative for",["lightblue"],["lightcoral"],   
                       "Apple",height=10)


# In[35]:


# Plot wordclouds for latest 1000 positive reviews for Oneplus
Op_pos_word_freq = word_freq_dict(Op_pos)
wordcloud = WordCloud(width=5000, 
                      height=3000, 
                      max_words=200, 
                      colormap="Blues",
                      background_color="white")
wordcloud_from_frequency(Op_pos_word_freq, "Most Frequent Words in the Latest 1000 Positive Reviews for Oneplus")


# In[36]:


# Plot wordclouds for latest 1000 negativereviews for Oneplus
Op_neg_word_freq = word_freq_dict(Op_neg)
wordcloud = WordCloud(width=5000, 
                      height=3000, 
                      max_words=200, 
                      colormap="Blues",
                      background_color="Black")
wordcloud_from_frequency(Op_neg_word_freq, "Most Frequent Words in the Latest 1000 Negative Reviews for Oneplus")


# In[37]:


#Plotting top 15 positive and negative words for Oneplus
Op_top_pos,Op_top_neg=topn_wordfreq_bar_both(Op_pos_word_freq, Op_neg_word_freq, 
                       min(sum(Op['Sentiment_Rating']=='Positive'), 1000), 
                       min(sum(Op['Sentiment_Rating']=='Negative'), 1000), 
                       15, 
                       "Top15 Frequent Words in Latest Positive for","Top15 Frequent Words in Latest Negative for",["lightblue"],["lightcoral"], 
                       "Oneplus",height=10)


# In[38]:


# Plot wordclouds for latest 1000 positive reviews for Vivo
Vivo_pos_word_freq = word_freq_dict(Vivo_pos)
wordcloud = WordCloud(width=5000, 
                      height=3000, 
                      max_words=200, 
                      colormap="Blues",
                      background_color="white")
wordcloud_from_frequency(Vivo_pos_word_freq, "Most Frequent Words in the Latest 1000 Positive Reviews for Vivo")


# In[39]:


# Plot wordclouds for latest 1000 negativereviews for Vivo
Vivo_neg_word_freq = word_freq_dict(Vivo_neg)
wordcloud = WordCloud(width=5000, 
                      height=3000, 
                      max_words=200, 
                      colormap="Blues",
                      background_color="Black")
wordcloud_from_frequency(Vivo_neg_word_freq, "Most Frequent Words in the Latest 1000 Negative Reviews for Vivo")


# In[40]:


#Plotting top 15 positive and negative words for Vivo
Vivo_top_pos,Vivo_top_neg=topn_wordfreq_bar_both(Vivo_pos_word_freq, Vivo_neg_word_freq, 
                       min(sum(Vivo['Sentiment_Rating']=='Positive'), 1000), 
                       min(sum(Vivo['Sentiment_Rating']=='Negative'), 1000), 
                       15, 
                       "Top15 Frequent Words in Latest Positive for","Top15 Frequent Words in Latest Negative for",["lightblue"],["lightcoral"],
                       "Vivo",height=10)


# In[41]:


# Plot wordclouds for latest 1000 positive reviews for Samsung
Sam_pos_word_freq = word_freq_dict(Sam_pos)
wordcloud = WordCloud(width=5000, 
                      height=3000, 
                      max_words=200, 
                      colormap="Blues",
                      background_color="white")
wordcloud_from_frequency(Sam_pos_word_freq, "Most Frequent Words in the Latest 1000 Positive Reviews for Samsung")


# In[42]:


# Plot wordclouds for latest 1000 negativereviews for Samsung
Sam_neg_word_freq = word_freq_dict(Sam_neg)
wordcloud = WordCloud(width=5000, 
                      height=3000, 
                      max_words=200, 
                      colormap="Blues",
                      background_color="Black")
wordcloud_from_frequency(Sam_neg_word_freq, "Most Frequent Words in the Latest 1000 Negative Reviews for Samsung")


# In[43]:


#Plotting top 15 positive and negative words for Samsung
Sam_top_pos,Sam_top_neg=topn_wordfreq_bar_both(Sam_pos_word_freq, Sam_neg_word_freq, 
                       min(sum(Samsung['Sentiment_Rating']=='Positive'), 1000), 
                       min(sum(Samsung['Sentiment_Rating']=='Negative'), 1000), 
                       15, 
                       "Top15 Frequent Words in Latest Positive for","Top15 Frequent Words in Latest Negative for",["lightblue"],["lightcoral"],
                       "Samsung",height=10)


# In[44]:


#finding common bigrams in top15 Positive and Negative reviews of All Companies.
common_values_neg = set.intersection(set(Op_top_neg['index']), set(Apple_top_neg['index']), set(Vivo_top_neg['index']), set(Sam_top_neg['index']))
common_values_pos=set.intersection(set(Op_top_pos['index']), set(Apple_top_pos['index']), set(Vivo_top_pos['index']), set(Sam_top_pos['index']))


# In[45]:


common_values_neg


# In[46]:


common_values_pos


# In[47]:


Common_bigrams_neg = pd.concat([Op_top_neg[Op_top_neg['index'].isin(common_values_neg)], Apple_top_neg[Apple_top_neg['index'].isin(common_values_neg)], Vivo_top_neg[Vivo_top_neg['index'].isin(common_values_neg)],Sam_top_neg[Sam_top_neg['index'].isin(common_values_neg)]], ignore_index=True)
Common_bigrams_pos = pd.concat([Op_top_pos[Op_top_pos['index'].isin(common_values_pos)], Apple_top_pos[Apple_top_pos['index'].isin(common_values_pos)], Vivo_top_pos[Vivo_top_pos['index'].isin(common_values_pos)],Sam_top_pos[Sam_top_pos['index'].isin(common_values_pos)]], ignore_index=True)


# In[48]:


Common_bigrams_neg.sort_values("frequency" ,axis = 0, ascending = False, 
                 inplace = True, na_position ='last')
Common_bigrams_neg


# In[49]:


Common_bigrams_pos.sort_values(["index","frequency"] ,axis = 0, ascending = False, 
                 inplace = True, na_position ='last')
Common_bigrams_pos


# In[50]:


#Common Negative Features 
 # nested barplot of common negative features of different companies.
ax = sns.barplot(x="Company", y="frequency", hue="index", data=Common_bigrams_neg)
plt.title("Frequency of common Negative features of different companies")


# In[51]:


#Common Negative Features 
 # nested barplot of common positive features of different companies.
ax = sns.barplot(x="Company", y="frequency", hue="index", data=Common_bigrams_pos)
plt.title("Frequency of common Positive features of different companies")


# # SENTIMENT ANALYSIS USING Naive Bayes

# In[52]:


#Covert Sentiment into Binary Classification of format('-1': Negative,'1' :Positive)
sent_dict = {'Positive':1, 'Negative':-1}
for key, value in sent_dict.items():
    result['Sentiment_Rating'] = result['Sentiment_Rating'].replace(key, value)
result


# In[53]:


#Create Bag of Words Model 
X=result['Comment']
y=result['Sentiment_Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
vectorizer=CountVectorizer()
BOW=vectorizer.fit_transform(X_train)
df=pd.DataFrame(BOW.toarray(),columns=vectorizer.get_feature_names())


# In[54]:


count=result['Sentiment_Rating'].value_counts()
print(count)


# In[56]:


from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(BOW, y_train)


# In[57]:


unique, counts = np.unique(y_train_res, return_counts=True)
print(list(zip(unique, counts)))


# In[58]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_res, y_train_res)

nb.score(X_train_res, y_train_res)


# In[59]:


X_test_vect = vectorizer.transform(X_test)

y_pred = nb.predict(X_test_vect)

y_pred


# In[60]:


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:




