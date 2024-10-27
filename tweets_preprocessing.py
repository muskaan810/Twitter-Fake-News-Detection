#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from langdetect import detect
from textblob import TextBlob
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter  import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from collections import Counter
import contractions
import spacy
from sklearn.model_selection import train_test_split
import joblib


# In[3]:


nltk.download('wordnet')


# In[4]:


from nltk.corpus import stopwords
nltk.download('stopwords')


# In[5]:


nltk.download('averaged_perceptron_tagger')


# In[7]:


mine = pd.read_excel("C://Users//Mushaan Khubchandani//Downloads//teslatweets 1 (1).xlsx")
mine.head()


# In[8]:


mine = mine.drop(['Unnamed: 0'], axis=1)
mine.shape


# In[9]:


fake = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//fake.csv")
fake.head()


# In[10]:


real = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//real.csv")
real.head()


# In[11]:


real['label'] = 1


# In[12]:


fake['label'] = 0


# In[13]:


tweets1 = fake.merge(real, how='outer')


# In[14]:


tweets1['topic'] = 1


# In[15]:


disaster_train = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//disaster_tweets//train.csv")
disaster_train.head(10)


# In[16]:


disaster = disaster_train.drop(['id','keyword', 'location'], axis=1)
disaster.columns = ['tweet', 'label']
disaster.head(10)


# In[17]:


disaster['topic'] = 2


# In[18]:


#tweets=disaster


# In[19]:


tweets2 = tweets1.merge(disaster, how='outer')


# In[20]:


covid_fake = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//misleading.csv")
covid_fake = covid_fake.drop(['Unnamed: 0', 'id'], axis=1)
covid_fake['label'] = 0
covid_fake.columns = ['tweet', 'label']
covid_fake.head(10)


# In[21]:


covid_real = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//genuine.csv")
covid_real = covid_real.drop(['Unnamed: 0', 'id'], axis=1)
covid_real['label'] = 1
covid_real.columns = ['tweet', 'label']
covid_real.head(10)


# In[22]:


tweets3 = covid_fake.merge(covid_real, how='outer')


# In[23]:


tweets3['topic'] = 3


# In[24]:


tweets = tweets2.merge(tweets3, how='outer')


# In[25]:


liar_train = pd.read_csv('C://Users//Mushaan Khubchandani//Downloads//liar_dataset//train.tsv',sep = '\t', header=None) 
liar_train = liar_train[[1,2]]
liar_train.head(10)


# In[26]:


liar_test = pd.read_csv('C://Users//Mushaan Khubchandani//Downloads//liar_dataset//test.tsv',sep = '\t', header=None) 
liar_test = liar_test[[1,2]]
liar_test.head(10)


# In[27]:


liar_val = pd.read_csv('C://Users//Mushaan Khubchandani//Downloads//liar_dataset//valid.tsv',sep = '\t', header=None) 
liar_val = liar_val[[1,2]]
liar_val.head(10)


# In[28]:


liar = [liar_train, liar_val, liar_test]
liar = pd.concat(liar)
liar.head()


# In[29]:


liar.columns = ['label', 'tweet']


# In[30]:


liar['label'].value_counts()


# In[31]:


liar['label'] = liar['label'].map({'false': 0, 'barely-true': 0, 'pants-fire': 0, 'mostly-true': 1,'true': 1,'half-true': 1})
liar['label'].value_counts()


# In[32]:


liar['topic'] = 4


# In[33]:


#tweets=liar


# In[34]:


tweets = tweets.merge(liar, how='outer')
tweets.info()


# In[35]:


finance = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//financial tweets//stockerbot-export.csv",  on_bad_lines='skip')
finance.head(10)


# In[36]:


finance.info()


# In[37]:


finance.describe(include='all')


# In[38]:


finance['verified'].value_counts()


# In[39]:


finance['label'] = pd.Series([1 if a == 'True' else 0 for a in finance['verified']])


# In[40]:


finances = finance[['text', 'label']]
finances.columns = ['tweet', 'label']
finances['topic'] = 5
finances.info()


# In[41]:


finances.head()


# In[42]:


finances.info()


# In[43]:


f1 = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//business_real_tweets.csv")
f1.head(10)


# In[44]:


f2 = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//business_real_tweets_2.csv")
f2.head(10)


# In[45]:


finances_new = pd.merge(f1, f2, how='outer')
finances_new.head(10)


# In[46]:


fnew = finances_new[['content']]
fnew['label'] = 1
fnew.columns = ['tweet', 'label']
finances_final = pd.merge(finances, fnew, how='outer')
finances_final.info()


# In[47]:


finances_final['label'].value_counts()


# In[48]:


finances_final['topic'] = 5


# In[49]:


tweets = tweets.merge(finances_final, how='outer')
tweets.info()


# In[50]:


others = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//others.csv")
others.head()


# In[51]:


others = others.drop(['Unnamed: 0'], axis=1)
others['label'].value_counts()


# In[52]:


others['topic'] = 6


# In[53]:


tweets = pd.merge(tweets, others, how='outer')
tweets.info()


# In[56]:


mine['topic'] = 8
tweets = pd.merge(tweets, mine, how='outer')


# In[57]:


sweet = tweets


# In[2756]:


sweet.to_csv("C://Users//Mushaan Khubchandani//Downloads//training_tweets.csv")


# In[2772]:


tweets = sweet


# In[58]:


tweets.describe(include='all')


# In[59]:


tweets['tweet'].isnull().sum()


# In[60]:


tweets = tweets.dropna(subset='tweet')


# In[61]:


tweets[tweets['tweet'].duplicated()]


# In[62]:


tweets.duplicated().sum()


# In[63]:


tweets = tweets.drop_duplicates()


# In[64]:


def count_urls(text):
    return len(re.findall(r'http\S+', text))

tweets['urls'] = tweets['tweet'].apply(count_urls)


# In[65]:


tweets['tweet'].duplicated().sum()


# In[66]:


tweets[tweets['tweet'].duplicated()]


# In[67]:


tweets= tweets.drop_duplicates(subset='tweet')


# In[68]:


tweets.head(10)


# In[69]:


def count_occurences(character, word_array):
    counter = 0
    for j, word in enumerate(word_array):
        for char in word:
            if char == character:
                counter += 1
    return counter

tweets['question_marks'] = tweets['tweet'].apply(lambda txt: count_occurences("?", txt)) 
tweets['exclamation_marks'] = tweets['tweet'].apply(lambda txt: count_occurences("!", txt)) 
tweets['hashtags'] = tweets['tweet'].apply(lambda txt: count_occurences("#", txt)) 
tweets['mentions'] = tweets['tweet'].apply(lambda txt: count_occurences("@", txt))


# In[70]:


def get_lowercase(tweet):
    text = tweet.lower()
    text = re.sub(r'https?:\/\/t\.co\/\S+', ' ', text)
    text = re.sub(r'\b@\w+', ' ', text)
    text = contractions.fix(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

tweets['tweet'] = tweets['tweet'].apply(get_lowercase)


# In[71]:


'''emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""'''


# In[72]:


'''regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)'''


# In[73]:


'''def tokenize(s):
    return tokens_re.findall(s)

def preproces(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

tweets['tokenized'] = tweets['tweet'].apply(preproces)'''


# In[74]:


#tweets['tweet'] = tweets['tweet'].dropna()


# In[75]:


tweets['tokens'] = tweets['tweet'].apply(nltk.word_tokenize)


# In[2065]:


'''def count_by_regex(regex,plain_text):
    return len(re.findall(regex,plain_text))

tweets['urls'] = tweets['tweet'].apply(lambda txt: count_by_regex("http.?://[^\s]+[\s]?",txt))'''


# In[76]:


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

    

tags = []
for i in tweets['tokens']:
    pos_tagged = nltk.pos_tag(i)  
    print(pos_tagged)
    
    tags.append(pos_tagged)


# In[77]:


def extract_named_entities(pos_tags):
    named_entities = []
    chunked = nltk.ne_chunk(pos_tags)
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            named_entities.append((entity, entity_type))
    return named_entities
tweets['pos_tags'] = tags
tweets['named_entities'] = tweets['pos_tags'].apply(extract_named_entities)


# In[78]:


def extract_pos_features(pos_tags):
    counter = Counter(tag for word, tag in pos_tags)
    prp = counter['PRP'] + counter['PRP$']  # Personal and possessive pronouns
    adj = counter['JJ'] + counter['JJR'] + counter['JJS']  # Adjectives
    noun = counter['NN'] + counter['NNS'] + counter['NNP'] + counter['NNPS']  # Nouns
    verb = counter['VB'] + counter['VBD'] + counter['VBG'] + counter['VBN'] + counter['VBP'] + counter['VBZ']  # Verbs
    return prp, adj, noun, verb

tweets['prp'], tweets['adj'], tweets['noun'], tweets['verb'] = zip(*tweets['pos_tags'].apply(extract_pos_features))


# In[79]:


tags_lemma=[]
for i in tags:
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), i))
    print(wordnet_tagged)
    tags_lemma.append(wordnet_tagged)


# In[80]:


lem = WordNetLemmatizer()


# In[81]:


pos_tag_lemmas=[]
for wordnet_tagged in tags_lemma:
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
        # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
        # else use the tag to lemmatize the token
            lemmatized_sentence.append(lem.lemmatize(word, tag))
    print(lemmatized_sentence)
    pos_tag_lemmas.append(lemmatized_sentence)


# In[82]:


tweets['tokens'] = pos_tag_lemmas


# In[83]:


tweets.head(10)


# In[84]:


tweets.tail(10)


# In[85]:


'''lem = WordNetLemmatizer()
tweets['tokens'] = tweets['tokens'].apply(lambda x: [lem.lemmatize(word) for word in x])'''


# In[86]:


'''stem = PorterStemmer()
tweets['tokens'] = tweets['tokens'].apply(lambda x: [stem.stem(word) for word in x])'''


# In[87]:


tweets.describe()


# In[88]:


tweets[tweets['label']==1].describe()


# In[89]:


tweets[tweets['label']==0].describe()


# In[90]:


tweets['tokens_joined'] = tweets['tokens'].str.join(' ')


# In[91]:


tweets.head(10)


# In[92]:


def remove_non_alphanumeric(tweet):
    pattern = re.compile('[^a-zA-Z\s]+')
    x = re.sub(pattern, ' ', tweet)
    return(x)

tweets['clean_tweets'] = tweets['tokens_joined'].apply(remove_non_alphanumeric)


# In[ ]:





# In[93]:


def language_detection(text):
    try:
        language = detect(text)
        return(language)
    except:
        return('error')
tweets['lang'] = tweets['tweet'].apply(language_detection)


# In[94]:


tweets['lang'].value_counts()


# In[95]:


tweets = tweets[tweets['lang'] == 'en']


# In[96]:


def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'

tweets['sentiment'] = tweets['tweet'].apply(get_sentiment)
tweets['sentiment'].value_counts()


# In[97]:


sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
tweets['sentiment'] = tweets['sentiment'].map(sentiment_mapping)


# In[98]:


tweets['topic'].value_counts()


# In[99]:


tweets['sentiment'].value_counts()


# In[100]:


tweets.head(10)


# In[2088]:


tweets[tweets['label']==0]['clean_tweets'].to_csv('C://Users//Mushaan Khubchandani//Downloads//fake_tweets.csv')  
faketweets = open('C://Users//Mushaan Khubchandani//Downloads//fake_tweets.csv').read()


# In[1875]:


wordcloud = WordCloud().generate(faketweets)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Fake News Word Cloud")
plt.show()


# In[ ]:


tweets[tweets['label']==1]['clean_tweets'].to_csv('C://Users//Mushaan Khubchandani//Downloads//real_tweets.csv')  
realtweets = open('C://Users//Mushaan Khubchandani//Downloads//real_tweets.csv').read()


# In[ ]:


wordcloud = WordCloud().generate(realtweets)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Real News Word Cloud")
plt.show()


# In[ ]:


pip install gensim


# In[2398]:


pip show gensim


# In[2399]:


pip show scipy


# In[2402]:


pip show scikit-learn


# In[2403]:


pip show numpy


# In[2405]:


pip install spacy


# In[2408]:


pip show spacy


# In[2410]:


get_ipython().system('python -m spacy download en_core_web_md')


# In[101]:


tweets['tokens'] = tweets['clean_tweets'].apply(lambda x: x.split())
stop_words = set(stopwords.words('english'))
tweets['new_tokens'] = tweets['tokens'].apply(lambda x : [w for w in x if w.lower() not in stop_words and len(w) > 1])


# In[102]:


nlp = spacy.load('en_core_web_md')  

def vectorize_tweet(tweet_tokens):
    vectors = [token.vector for token in nlp(' '.join(tweet_tokens))]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(nlp.vocab.vectors_length)  


# In[103]:


# Example usage assuming `tweets` DataFrame and `new_tokens` column already exist
tweets['vector'] = tweets['new_tokens'].apply(vectorize_tweet)


# In[104]:


# Remove rows where vectorization failed (optional)
tweets = tweets[tweets['vector'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]


# In[105]:


# Prepare X and y for classification
X = np.hstack((
    np.stack(tweets['vector'].values),
    tweets[['urls', 'question_marks', 'exclamation_marks', 'hashtags', 'mentions', 'sentiment', 'prp', 'adj', 'noun', 'verb']].values
))
y = tweets['label'].values


# In[ ]:


from gensim.models import Word2Vec


# In[2267]:





# In[2268]:


w2v_model = Word2Vec(sentences=tweets['new_tokens'], vector_size=100, window=5, min_count=1, workers=4)


# In[2397]:


# Save the model
joblib.dump(w2v_model, "C://Users//Mushaan Khubchandani//OneDrive - Verolt Engineering Pvt Ltd//Documents//FakeNews//word2vec_model.model")


# In[2269]:


def vectorize_tweet(tweet_tokens):
    vector = np.mean([w2v_model.wv[word] for word in tweet_tokens if word in w2v_model.wv], axis=0)
    if isinstance(vector, np.ndarray):
        return vector
    else:
        return np.zeros(w2v_model.vector_size)


# In[2270]:


tweets['vector'] = tweets['new_tokens'].apply(vectorize_tweet)


# In[2271]:


tweets = tweets[tweets['vector'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]


# In[ ]:


'''sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
tweets['sentiment'] = tweets['sentiment'].map(sentiment_mapping)'''


# In[2382]:


X = np.hstack((
    np.stack(tweets['vector'].values),
    tweets[['urls', 'question_marks', 'exclamation_marks', 'hashtags', 'mentions', 'sentiment', 'prp', 'adj', 'noun', 'verb']].values
))
y = tweets['label'].values


# In[2385]:


pip install imblearn


# In[106]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[107]:


print(y_test)


# In[108]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize the model
classifier = LogisticRegression(max_iter=1000)

# Train the model
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

acc_log = round(classifier.score(X_train, y_train) * 100, 2)
print(acc_log)


# In[109]:


joblib.dump(classifier, 'C://Users//Mushaan Khubchandani//OneDrive - Verolt Engineering Pvt Ltd//Documents//FakeNews//logistic_model_with_mine.joblib')


# In[110]:


tweets.head(10)


# In[111]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Create the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Classifier Report")
print(classification_report(y_test, y_pred_rf))
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')


# In[112]:


joblib.dump(rf_model, 'C://Users//Mushaan Khubchandani//OneDrive - Verolt Engineering Pvt Ltd//Documents//FakeNews//rf_model_with_mine.joblib')


# In[2098]:


new_clean_tweets = ["You should have sold #BTC when it was at 73M.  You have lost 20%.  How much more are you going to lose?  #BTC will continue to fall.  Your stupidity is killing you."]


# In[2099]:


new_clean_tweets = pd.DataFrame(new_clean_tweets)


# In[2100]:


print(new_clean_tweets)


# In[2101]:


tokenized = new_clean_tweets[0].apply(nltk.word_tokenize)


# In[2102]:


tokenized = tokenized.apply(lambda x : [w for w in x if w.lower() not in stop_words])  


# In[38]:


tokenized.apply(lambda x: x.len())


# In[2103]:


tags = []
for i in tokenized:
    pos_tagged = nltk.pos_tag(i)  
    print(pos_tagged)
    tags.append(pos_tagged)


# In[2104]:


tags_lemma=[]
for i in tags:
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), i))
    print(wordnet_tagged)
    tags_lemma.append(wordnet_tagged)


# In[2105]:


pos_tag_lemmas=[]
for wordnet_tagged in tags_lemma:
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
        # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
        # else use the tag to lemmatize the token
            lemmatized_sentence.append(lem.lemmatize(word, tag))
    print(lemmatized_sentence)
    pos_tag_lemmas.append(lemmatized_sentence)


# In[2106]:


new_predict=pd.DataFrame()
new_predict[0] = pos_tag_lemmas


# In[2107]:


new_tokens = new_predict[0].str.join(' ')


# In[2108]:


new_tokens = new_tokens.apply(remove_non_alphanumeric)
print(new_tokens)


# In[2109]:


# Tokenize the new_clean_tweets
new_tokens = [tweet.split() for tweet in new_tokens]


# In[2110]:


new_vectors = np.array([vectorize_tweet(tokens) for tokens in new_tokens])


# In[2112]:


sentiment = new_clean_tweets[0].apply(get_sentiment)
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
sentiment = pd.DataFrame(sentiment)[0].map(sentiment_mapping)


# In[2113]:


print(sentiment)


# In[2114]:


qm = new_clean_tweets[0].apply(lambda txt: count_occurences("?", txt)) 
em = new_clean_tweets[0].apply(lambda txt: count_occurences("!", txt)) 
hashtags = new_clean_tweets[0].apply(lambda txt: count_occurences("#", txt)) 
mentions = new_clean_tweets[0].apply(lambda txt: count_occurences("@", txt))
urls = new_clean_tweets[0].apply(lambda txt: count_by_regex("http.?://[^\s]+[\s]?",txt))


# In[2115]:


topic = 5


# In[2116]:


qm = qm.values.reshape(-1, 1)
em = em.values.reshape(-1, 1)
hashtags = hashtags.values.reshape(-1, 1)
mentions = mentions.values.reshape(-1, 1)
urls = urls.values.reshape(-1, 1)


# In[1667]:


#sentiment = sentiment.values.reshape(-1, 1)


# In[2117]:


print(sentiment)
print([[topic]])


# In[2118]:


predict_features = np.hstack((
    new_vectors,
    urls,
    qm,
    em,
    hashtags,
    mentions, [[topic]]
))


# In[2119]:


new_predictions = classifier.predict(predict_features)

# Map predictions to readable format
prediction_labels = ['Fake' if label == 0 else 'Real' for label in new_predictions]

# Output the results
for tweet, prediction in zip(new_clean_tweets, prediction_labels):
    print(f'Tweet: {tweet} -> Prediction: {prediction}')


# In[2598]:


pip install contractions


# ## Testing Tweets

# In[188]:


test_tweets = pd.read_csv("C://Users//Mushaan Khubchandani//OneDrive - Verolt Engineering Pvt Ltd//Documents//FakeNews//Tweets_Scraped//all_tweets_full_final.csv")
test_tweets = test_tweets.drop(['Unnamed: 0'], axis=1)
test_tweets.shape


# In[157]:


test_tweets['temp'] = pd.to_datetime(test_tweets['date'], errors='coerce')
invalid_dates = test_tweets[test_tweets['date'].isna()]
if not invalid_dates.empty:
    print("There are invalid date formats in the DataFrame:")
    print(invalid_dates)
else:
    print("All dates are valid.")


# In[143]:


test_tweets= test_tweets[test_tweets['temp'].dt.date == pd.to_datetime('2024-07-05').date()]
test_tweets = test_tweets.drop(['temp'], axis=1)


# In[144]:


new_dfnew_df.drop_duplicates(subset='content').shape


# In[145]:


#df = test_tweets.sort_values(by=['likes', 'retweets'], ascending=False)


# In[146]:


#df['content'].iloc[:100].to_excel("C://Users//Mushaan Khubchandani//Downloads//teslatweets.xlsx")


# In[147]:


test_tweets['content']


# In[158]:


'''def language_detection(text):
    try:
        language = detect(text)
        return(language)
    except:
        return('error')
test_tweets['lang'] = test_tweets['content'].apply(language_detection)'''
print(test_tweets['lang'].value_counts())


# In[189]:


def count_urls(text):
    return len(re.findall(r'http\S+', text))

test_tweets['urls'] = test_tweets['content'].apply(count_urls)

def count_occurences(character, word_array):
    counter = 0
    for j, word in enumerate(word_array):
        for char in word:
            if char == character:
                counter += 1
    return counter

test_tweets['qm'] = test_tweets['content'].apply(lambda txt: count_occurences("?", txt)) 
test_tweets['em'] = test_tweets['content'].apply(lambda txt: count_occurences("!", txt)) 
test_tweets['num_hashtags'] = test_tweets['content'].apply(lambda txt: count_occurences("#", txt)) 
test_tweets['num_mentions'] = test_tweets['content'].apply(lambda txt: count_occurences("@", txt))

import contractions
def get_lowercase(tweet):
    text = tweet.lower()
    text = re.sub(r'https?:\/\/t\.co\/\S+', ' ', text)
    text = re.sub(r'\b@\w+', ' ', text)
    text = contractions.fix(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

test_tweets['clean_content'] = test_tweets['content'].apply(get_lowercase)
test_tweets.info()


# In[190]:


test_tweets['tokenized'] = test_tweets['clean_content'].apply(nltk.word_tokenize)
test_tweets.info()


# In[191]:


test_tweets['len'] = test_tweets['tokenized'].apply(len)
test_tweets = test_tweets[test_tweets['len'] > 3]
test_tweets = test_tweets.drop(['len'], axis=1)
test_tweets.info()


# In[192]:


test_tweets.head(10)


# In[193]:


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

tags = []
for i in test_tweets['tokenized']:
    pos_tagged = nltk.pos_tag(i)  
    tags.append(pos_tagged)
    
    
def extract_named_entities(pos_tags):
    named_entities = []
    chunked = nltk.ne_chunk(pos_tags)
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            named_entities.append((entity, entity_type))
    return named_entities
test_tweets['pos_tags'] = tags
test_tweets['named_entities'] = test_tweets['pos_tags'].apply(extract_named_entities)

from collections import Counter

def extract_pos_features(pos_tags):
    counter = Counter(tag for word, tag in pos_tags)
    prp = counter['PRP'] + counter['PRP$']  # Personal and possessive pronouns
    adj = counter['JJ'] + counter['JJR'] + counter['JJS']  # Adjectives
    noun = counter['NN'] + counter['NNS'] + counter['NNP'] + counter['NNPS']  # Nouns
    verb = counter['VB'] + counter['VBD'] + counter['VBG'] + counter['VBN'] + counter['VBP'] + counter['VBZ']  # Verbs
    return prp, adj, noun, verb

test_tweets['prp'], test_tweets['adj'], test_tweets['noun'], test_tweets['verb'] = zip(*test_tweets['pos_tags'].apply(extract_pos_features))

tags_lemma=[]
for i in tags:
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), i))
    tags_lemma.append(wordnet_tagged)
    
pos_tag_lemmas=[]
lem = WordNetLemmatizer()
for wordnet_tagged in tags_lemma:
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
        # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
        # else use the tag to lemmatize the token
            lemmatized_sentence.append(lem.lemmatize(word, tag))
    pos_tag_lemmas.append(lemmatized_sentence)
    
test_tweets['lemmas'] = pos_tag_lemmas
test_tweets['lemmas'] = test_tweets['lemmas'].str.join(' ')

def remove_non_alphanumeric(tweet):
    pattern = re.compile('[^a-zA-Z\s]+')
    x = re.sub(pattern, ' ', tweet)
    return(x)

test_tweets['lemmas'] = test_tweets['lemmas'].apply(remove_non_alphanumeric)



test_tweets['tokens'] = [tweet.split() for tweet in test_tweets['lemmas']]
stop_words = set(stopwords.words('english'))
test_tweets['tokens'] = test_tweets['tokens'].apply(lambda x : [w for w in x if w.lower() not in stop_words and len(w) > 1]) 
test_tweets.info()


# In[194]:


# Load spaCy model with pre-trained word vectors
nlp = spacy.load('en_core_web_md')  # Use 'en_core_web_lg' for larger vectors if needed

# Vectorization function using spaCy's pre-trained embeddings
def vectorize_tweet(tweet_tokens):
    vectors = [token.vector for token in nlp(' '.join(tweet_tokens))]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(nlp.vocab.vectors_length)  # Adjust this based on vector size in the loaded model
vectors = np.array([vectorize_tweet(tokens) for tokens in test_tweets['tokens']])
test_tweets.info()


# In[195]:


def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'

test_tweets['sentiment'] = test_tweets['content'].apply(get_sentiment)
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
test_tweets['sentiment'] = test_tweets['sentiment'].map(sentiment_mapping)
test_tweets.info()


# In[196]:


import joblib
predict_features = np.hstack((
    np.stack(vectors),
    test_tweets[['urls', 'qm', 'em', 'num_hashtags', 'num_mentions', 'sentiment', 'prp', 'adj', 'noun', 'verb']].values
))
classifier = joblib.load("C://Users//Mushaan Khubchandani//OneDrive - Verolt Engineering Pvt Ltd//Documents//FakeNews//rf_model_with_mine.joblib")
new_predictions = classifier.predict(predict_features)

# Map predictions to readable format
test_tweets['prediction'] = ['Fake' if label == 0 else 'Real' for label in new_predictions]


# Output the results
test_tweets.info()


# In[197]:


test_tweets['prediction'].value_counts()


# In[198]:


test_tweets['sentiment'].value_counts()


# In[185]:


test_tweets = test_tweets.sort_values(by=['likes', 'retweets'], ascending=False)


# In[186]:


fake_contents = test_tweets.loc[test_tweets['prediction'] == 'Fake', 'content']
print("Fake contents:")
for tweet in fake_contents:
    print('--> ', tweet)
    print()


# In[187]:


real_contents = test_tweets.loc[test_tweets['prediction'] == 'Real', 'content']
print("Real Tweets:")
for tweet in real_contents:
    print('--> ', tweet)
    print()


# In[167]:


test_tweets.tail()


# In[168]:


test_tweets.tail(10)


# In[169]:


test_tweets['date'][0]


# In[170]:


test_tweets['datetime'] = pd.to_datetime(test_tweets['date'])


# In[171]:


test_tweets['date'] = test_tweets['datetime'].dt.date
test_tweets['time'] = test_tweets['datetime'].dt.time


# In[172]:


test_tweets.head()


# In[88]:


#to_fix = test_tweets[test_tweets['time'] > pd.to_datetime('20:00:00').time()]


# In[77]:


#to_fix['hashtags'].value_counts()


# In[80]:


#mask = to_fix['content'].str.contains('@Tesla|#Tesla', case=False, regex=True)
#fixed = to_fix[mask]


# In[89]:


#other = test_tweets[test_tweets['time'] < pd.to_datetime('20:00:00').time()]


# In[140]:


test_tweets.info()


# In[93]:


#other.info()


# In[96]:


#tesla = pd.concat([other,fixed])


# In[201]:


#tesla = test_tweets.drop(['tokens', 'lemmas', 'pos_tags', 'tokenized', 'clean_content'], axis=1)
test_tweets.to_csv("C://Users//Mushaan Khubchandani//OneDrive - Verolt Engineering Pvt Ltd//Documents//FakeNews//Tweets_Scraped//all_tweets_rf_final.csv")


# In[201]:


tt = pd.read_csv("C://Users//Mushaan Khubchandani//OneDrive - Verolt Engineering Pvt Ltd//Documents//FakeNews//Tweets_Scraped//all_tweets.csv")
tt = tt.drop(['Unnamed: 0'], axis=1)
tt.tail(10)


# In[208]:


new_df['company'].value_counts()


# In[133]:


test_tweets['verified'] = test_tweets['verified'].map({'False': False, 'True': True}).astype('bool')


# In[197]:


test_tweets['datetime'] = pd.to_datetime(test_tweets['datetime'], utc=True)


# In[198]:


test_tweets['datetime'] = test_tweets['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
test_tweets['datetime'] = test_tweets['datetime'].str[:22] + ':' + test_tweets['datetime'].str[22:]


# In[199]:


test_tweets['datetime']


# In[202]:


new_df = pd.concat([test_tweets, tt])


# In[147]:


tes = pd.read_csv("C://Users//Mushaan Khubchandani//Downloads//tesla07-05-5pm.csv", header=None)
tes = tes.drop([0], axis=1)
tes.columns =['date','id','content','username','lang','likes','retweets','replies','hashtags','location','followers_count','friends_count','profile_image_url','verified']
tes['company'] = 'Tesla'
tes.tail(10)


# In[148]:


tes.info()


# In[10]:


test_tweets.iloc[3000:3010]


# In[173]:


import googlemaps
import time
import pandas as pd

# Replace 'YOUR_API_KEY' with your actual Google Maps API key
gmaps = googlemaps.Client(key='AIzaSyAariU9HXksneRtRh6cf3hbRiKBHcdgQWs')


# In[179]:


tt = test_tweets['location'].fillna('0')


# In[180]:


test_tweets['loctp'] = tt


# In[181]:


locs = test_tweets[test_tweets['loctp'] != '0']['loctp']


# In[182]:


locations = locs.tolist()


# In[183]:


batch_size = 200
results = []

# Iterate through locations in batches
for i in range(0, len(locations), batch_size):
    geocode_batch = locations[i:i+batch_size]  # Get a batch of locations
    
    for location in geocode_batch:
        try:
            geocode_result = gmaps.geocode(location)  # Send geocode request
            if geocode_result:
                # Extract latitude and longitude from successful results
                lat = geocode_result[0]['geometry']['location']['lat']
                lng = geocode_result[0]['geometry']['location']['lng']
                results.append({'location': location, 'latitude': lat, 'longitude': lng})  # Append coordinates to results list
            else:
                print(f"Geocoding failed for location: {location}")
                results.append({'location': location, 'latitude': None, 'longitude': None})
        except Exception as e:
            print(f"Error geocoding {location}: {e}")
            results.append({'location': location, 'latitude': None, 'longitude': None})
        
        # Introduce a small delay to avoid exceeding QPS limit
        time.sleep(0.1)  # Adjust delay as needed

# Convert results to DataFrame
results_df = pd.DataFrame(results)
unique_locations = results_df.groupby('location').agg({'latitude': 'first', 'longitude': 'first'}).reset_index()


# In[184]:


test_tweets.info()


# In[185]:


test_tweets = test_tweets.set_index('location').combine_first(unique_locations.set_index('location')).reset_index()


# In[107]:


pip install geopy


# In[186]:


test_tweets= test_tweets.drop(['loctp'], axis=1)


# In[189]:


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def get_country(lat, lon):
    try:
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.reverse((lat, lon), timeout=10)
        if location is not None:
            address = location.raw.get('address', {})
            country = address.get('country', 'Unknown')
            return country
        else:
            return 'Unknown'
    except GeocoderTimedOut:
        return 'Unknown'
    except Exception:
        return 'Unknown'

df_filtered = test_tweets.dropna(subset=['latitude', 'longitude'])

df_filtered['country'] = df_filtered.apply(lambda row: get_country(row['latitude'], row['longitude']), axis=1)


# In[190]:


print(df_filtered['country'].value_counts().index)


# In[191]:


# Standardize country names
country_name_mapping = {
    'Deutschland': 'Germany',
    'Italia': 'Italy',
    'مصر': 'Egypt',
    'Sverige': 'Sweden',
    'Schweiz/Suisse/Svizzera/Svizra': 'Switzerland',
    'الإمارات العربية المتحدة': 'United Arab Emirates',
    '中国': 'China',
    'Norge': 'Norway',
    'España': 'Spain',
    '日本': 'Japan',
    'السعودية': 'Saudi Arabia',
    'Suomi / Finland': 'Finland',
    'México': 'Mexico',
    '대한민국': 'South Korea',
    'Danmark': 'Denmark',
    'Österreich': 'Austria',
    'Lëtzebuerg': 'Luxembourg',
    'Polska': 'Poland',
    'Éire / Ireland': 'Ireland',
    '臺灣': 'Taiwan',
    'Ελλάς': 'Greece',
    'România': 'Romania',
    'پاکستان': 'Pakistan',
    'বাংলাদেশ': 'Bangladesh',
    'Maroc ⵍⵎⵖⵔⵉⴱ المغرب': 'Morocco',
    'ישראל': 'Israel',
    'საქართველო': 'Georgia',
    'Latvija': 'Latvia',
    'Tanzania': 'Tanzania',
    'België / Belgique / Belgien': 'Belgium',
    'Việt Nam': 'Vietnam',
    'Česko': 'Czech Republic',
    'Україна': 'Ukraine',
    'Pilipinas': 'Philippines',
    'ประเทศไทย': 'Thailand',
    'Ísland': 'Iceland',
    'Türkiye': 'Turkey',
    'افغانستان': 'Afghanistan',
    'Северна Македонија': 'North Macedonia',
    'Magyarország': 'Hungary',
    'България': 'Bulgaria',
    'الأردن': 'Jordan',
    'Hrvatska': 'Croatia',
    'ایران': 'Iran',
    'República Dominicana': 'Dominican Republic',
    'Lietuva': 'Lithuania',
    'Қазақстан': 'Kazakhstan',
    'ປະເທດລາວ': 'Laos',
    'Eesti': 'Estonia',
    'United States': 'United States',
    'United Kingdom': 'United Kingdom',
    'Canada': 'Canada',
    'Deutschland': 'Germany',
    'India': 'India',
    'ישראל': 'Israel',
    'België / Belgique / Belgien': 'Belgium',
    'South Africa': 'South Africa',
    'France': 'France',
    'Italia': 'Italy',
    'Schweiz/Suisse/Svizzera/Svizra': 'Switzerland',
    'Sverige': 'Sweden',
    'Singapore': 'Singapore',
    'Slovenija': 'Slovenia',
    'México': 'Mexico',
    'Österreich': 'Austria',
    'Nigeria': 'Nigeria',
    'Kenya': 'Kenya',
    'नेपाल': 'Nepal',
    'Costa Rica': 'Costa Rica',
    'България': 'Bulgaria',
    'Éire / Ireland': 'Ireland',
    'Portugal': 'Portugal',
    'România': 'Romania',
    'New Zealand / Aotearoa': 'New Zealand',
    'السعودية': 'Saudi Arabia', 
     'Unknown': 'Unknown',
    'United States': 'United States',
    'Canada': 'Canada',
    'Australia': 'Australia',
    'Singapore': 'Singapore',
    'India': 'India',
    'France': 'France',
    'United Kingdom': 'United Kingdom',
    'Sverige': 'Sweden',
    'Norge': 'Norway',
    '中国': 'China',
    'Nederland': 'Netherlands',
    '대한민국': 'South Korea',
    'Česko': 'Czech Republic',
    'Deutschland': 'Germany',
    'Österreich': 'Austria',
    'Schweiz/Suisse/Svizzera/Svizra': 'Switzerland',
    'Malaysia': 'Malaysia',
    'Polska': 'Poland',
    'España': 'Spain',
    '臺灣': 'Taiwan',
    'Italia': 'Italy',
    'Portugal': 'Portugal',
    'Nigeria': 'Nigeria',
    'South Africa': 'South Africa',
    'New Zealand / Aotearoa': 'New Zealand',
    'Україна': 'Ukraine',
    '日本': 'Japan',
    'México': 'Mexico',
    'Việt Nam': 'Vietnam',
    'Moçambique': 'Mozambique',
    'ישראל': 'Israel',
    'България': 'Bulgaria',
    'عمان': 'Oman',
    'Éire / Ireland': 'Ireland',
    'Magyarország': 'Hungary',
    'Malawi': 'Malawi',
    'Danmark': 'Denmark',
    'Kenya': 'Kenya',
    'Slovenija': 'Slovenia',
    '조선민주주의인민공화국': 'North Korea',
    'Chile': 'Chile'
}

df_filtered['country'] = df_filtered['country'].replace(country_name_mapping)


# In[192]:


df_filtered.info()


# In[193]:


df = df_filtered[['location', 'country']]


# In[194]:


location_to_country = df.set_index('location')['country'].to_dict()


# In[195]:


# Map the country values to the test_tweets DataFrame
test_tweets['country'] = test_tweets['location'].map(location_to_country).fillna('Unknown')


# In[196]:


test_tweets.info()


# In[110]:





# In[98]:


test_tweets.merge(unique_locations, on=['location', 'latitude', 'longitude'], how='outer').info()


# In[32]:


test_tweets['lemmas'].to_csv('C://Users//Mushaan Khubchandani//Downloads//nvidia_words.csv')  
nvidiawords = open('C://Users//Mushaan Khubchandani//Downloads//nvidia_words.csv').read()


# In[116]:


from datetime import datetime

# Assuming your timestamp object is stored in a variable named 'timestamp_obj'
timestamp_obj = pd.Timestamp('2024-07-06 23:59:18+0000', tz='UTC')

# Convert to the desired format
formatted_datetime = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')
print(formatted_datetime)

test_tweets[test_tweets['datetime'].apply(strftime('%Y-%m-%d %H:%M:%S')) == datetime.datetime(2024, 7, 5)]


# In[33]:


wordcloud = WordCloud().generate(nvidiawords)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Fake News Word Cloud")
plt.show()


# In[121]:


test_tweets['tokens']


# In[120]:


all_words = [word for tokens in test_tweets['tokens'] for word in tokens]

# Count the occurrences of each word
word_counts = Counter(all_words)

# Create a DataFrame from the word counts
word_counts_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])

# Sort the DataFrame in descending order of count
word_counts_df = word_counts_df.sort_values(by='count', ascending=False).reset_index(drop=True)

word_counts_df.head(20)


# In[3139]:


test_tweets[test_tweets['sentiment'] == -1]['content'][2]


# In[199]:


probabs = classifier.predict_proba(predict_features)
print(probabs)


# In[200]:


predicted_labels=[]
for i in probabs:
    if i[0] > 0.50:
        if i[0] >= 0.75:
            predicted_labels.append('Fake')
        else:
            predicted_labels.append('Opinion')
    else:
        predicted_labels.append('Real')
    #labels = ['Fake' if i[0] > 0.85 else 'Real']
    #predicted_labels.append(labels[0])
    
test_tweets['predicted_labels'] = predicted_labels
test_tweets.info()


# In[ ]:





# In[137]:


test_tweets['predicted_labels'].value_counts()


# In[138]:


fake_indices = test_tweets[test_tweets['predicted_labels'] == 'Opinion']['content'].index
for i in fake_indices:
    print('-->', test_tweets.iloc[i]['content'])


# In[178]:


real_indices = test_tweets[test_tweets['predicted_labels'] == 'Real']['content'].index
for i in real_indices:
    print('-->', test_tweets.iloc[i]['content'])


# In[3025]:


real_indices = test_tweets[test_tweets['predicted_labels'] == 'cannot be determined']['content'].index
for i in real_indices:
    print('-->', test_tweets.iloc[i]['content'])


# In[1]:


test_tweets['content']


# In[2708]:


labels = tweets['label'].values


# In[404]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# In[405]:


import torch
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# In[406]:


import torch.nn as nn
import torch.optim as optim

class TweetClassifierWithMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(TweetClassifierWithMemory, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = 2
n_layers = 1

model = TweetClassifierWithMemory(input_dim, hidden_dim, output_dim, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[407]:


# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    hidden = (torch.zeros(n_layers, X_train_tensor.size(0), hidden_dim),
              torch.zeros(n_layers, X_train_tensor.size(0), hidden_dim))
    
    outputs, hidden = model(X_train_tensor.unsqueeze(1), hidden)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# In[65]:


model.eval()
with torch.no_grad():
    hidden = (torch.zeros(n_layers, X_test_tensor.size(0), hidden_dim),
              torch.zeros(n_layers, X_test_tensor.size(0), hidden_dim))
    outputs, hidden = model(X_test_tensor.unsqueeze(1), hidden)
    test_loss = criterion(outputs, y_test_tensor)
    predictions = torch.argmax(outputs, axis=1)
    accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)

print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy*100:.2f}%')


# In[ ]:


######


# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout

EMBEDDING_DIM = 100

model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

