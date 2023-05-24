import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

def pre_process_tweet(tweet):
    #converting to lowercase
    tweet = tweet.lower()

    #removing hyperlinks
    tweet = re.sub(r'http\S+', '', tweet)

    #removing username from text
    tweet = re.sub(r'@\w+','', tweet)

    #removing hashtags from text
    tweet = re.sub(r'#','',tweet)

    #removing digits
    tweet = re.sub(r'\d+', '', tweet)

    #removing punctuations
    tweet = re.sub(r'[^\w\s]|_', '', tweet)

    # converting to tokens
    tweet_tokens = word_tokenize(tweet)

    # removing stop words
    stop_words = set(stopwords.words('english'))
    cleaned_tweet_tokens = [word for word in tweet_tokens if word not in stop_words ]

    # stemming
    # stemmer = PorterStemmer()
    # stemmed_tweets = [stemmer.stem(word) for word in cleaned_tweet_tokens]

    return cleaned_tweet_tokens
