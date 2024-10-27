from collections import Counter
import re
import nltk
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from langdetect import detect
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import spacy
import contractions


nlp = spacy.load('en_core_web_md')

import joblib

nltk.download('wordnet')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger')

model = joblib.load('logistic_model.joblib')  


def language_detection(text):
    try:
        language = detect(text)
        return(language)
    except:
        return('error')


def count_urls(text):
    return len(re.findall(r'http\S+', text))  

def get_lowercase(tweet):
    text = tweet.lower()
    text = re.sub(r'https?:\/\/t\.co\/\S+', ' ', text)
    text = re.sub(r'\b@\w+', ' ', text)
    text = contractions.fix(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_occurences(character, word_array):
    counter = 0
    for j, word in enumerate(word_array):
        for char in word:
            if char == character:
                counter += 1
    return counter

def extract_pos_features(pos_tags):
    counter = Counter(tag for word, tag in pos_tags)
    prp = counter['PRP'] + counter['PRP$']  # Personal and possessive pronouns
    adj = counter['JJ'] + counter['JJR'] + counter['JJS']  # Adjectives
    noun = counter['NN'] + counter['NNS'] + counter['NNP'] + counter['NNPS']  # Nouns
    verb = counter['VB'] + counter['VBD'] + counter['VBG'] + counter['VBN'] + counter['VBP'] + counter['VBZ']  # Verbs
    return prp, adj, noun, verb

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
   
def extract_named_entities(pos_tags):
    named_entities = []
    chunked = nltk.ne_chunk(pos_tags)
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            named_entities.append((entity, entity_type))
    return named_entities

def remove_non_alphanumeric(tweet):
        pattern = re.compile('[^a-zA-Z\\s]+')
        x = re.sub(pattern, ' ', tweet)
        return(x)

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'



def vectorize_tweet(tweet_tokens):
    vectors = [token.vector for token in nlp(' '.join(tweet_tokens))]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(nlp.vocab.vectors_length)
    

def preprocess_tweet(test_tweets):
    test_tweets = test_tweets.drop_duplicates(subset='content')

    test_tweets['urls'] = test_tweets['content'].apply(count_urls)

    test_tweets['qm'] = test_tweets['content'].apply(lambda txt: count_occurences("?", txt)) 
    test_tweets['em'] = test_tweets['content'].apply(lambda txt: count_occurences("!", txt)) 
    test_tweets['hashtags'] = test_tweets['content'].apply(lambda txt: count_occurences("#", txt)) 
    test_tweets['mentions'] = test_tweets['content'].apply(lambda txt: count_occurences("@", txt))

    test_tweets['clean_content'] = test_tweets['content'].apply(get_lowercase)

    test_tweets['tokenized'] = test_tweets['clean_content'].apply(nltk.word_tokenize)
   
    tags = []
    for i in test_tweets['tokenized']:
        pos_tagged = nltk.pos_tag(i)  
        tags.append(pos_tagged)
        
    test_tweets['pos_tags'] = tags
    test_tweets['named_entities'] = test_tweets['pos_tags'].apply(extract_named_entities)

    test_tweets['prp'], test_tweets['adj'], test_tweets['noun'], test_tweets['verb'] = zip(*test_tweets['pos_tags'].apply(extract_pos_features))

    tags_lemma=[]
    for i in tags:
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), i))
        tags_lemma.append(wordnet_tagged)

    lem = WordNetLemmatizer()
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
        pos_tag_lemmas.append(lemmatized_sentence)
        
    test_tweets['lemmas'] = pos_tag_lemmas
    test_tweets['lemmas'] = test_tweets['lemmas'].str.join(' ')

    test_tweets['lemmas'] = test_tweets['lemmas'].apply(remove_non_alphanumeric)

    test_tweets['tokens'] = [tweet.split() for tweet in test_tweets['lemmas']]

    test_tweets['tokens'] = test_tweets['tokens'].apply(lambda x : [w for w in x if w.lower() not in stop_words and len(w) > 1]) 

    #w2v_model = Word2Vec(sentences=test_tweets['new_tokens'], vector_size=100, window=5, min_count=1, workers=4)
    vectors = np.array([vectorize_tweet(tokens) for tokens in test_tweets['tokens']])

    test_tweets['sentiment'] = test_tweets['content'].apply(get_sentiment)
    sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
    test_tweets['sentiment'] = test_tweets['sentiment'].map(sentiment_mapping)
    
    test_tweets['datetime'] = pd.to_datetime(test_tweets['date'])
    test_tweets['date'] = test_tweets['datetime'].dt.date
    test_tweets['time'] = test_tweets['datetime'].dt.time

    predict_features = np.hstack((
        np.stack(vectors),
        test_tweets[['urls', 'qm', 'em', 'hashtags', 'mentions', 'sentiment', 'prp', 'adj', 'noun', 'verb']].values
    ))

    new_predictions = model.predict(predict_features)

    test_tweets['prediction'] = ['Fake' if label == 0 else 'Real' for label in new_predictions]
    
    return test_tweets

def classify_tweets(model, tweets):
    return model.predict(tweets)
