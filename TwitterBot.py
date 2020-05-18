#!/usr/bin/python3

# Import Modules
import pandas as pd
import numpy as np
import datetime as dt
import pyspark
import time
import lyricsgenius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import billboard
import time
import requests
import tweepy
import re
import GetOldTweets3 as got
import nltk
import gcsfs
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.tokenize import word_tokenize
from string import punctuation

from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import CountVectorizer , IDF
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import BinaryClassificationMetrics

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
from gensim import corpora, models
from pprint import pprint
from collections import Counter

APP_NAME = "Random Forest"
SPARK_URL = "local[*]"

spark = SparkSession.builder \
    .appName(APP_NAME) \
    .master(SPARK_URL) \
    .getOrCreate()

export SPOTIPY_CLIENT_ID=28aa32e2bfbc4a2183b24afc6e52a40c
export SPOTIPY_CLIENT_SECRET=48c3512efc3f4098bdf80304e0a67bab

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

for x in range(5):
    try:
        trackids = []
        for val in sp.playlist_tracks('37i9dQZF1DWUa8ZRTfalHk', fields='items').values():
            for x in val:
                for key, res in x.items():
                    if(key == 'track'):
                        for x, vals in res.items():
                            if x == 'id':
                                trackids.append(vals)
        for val in sp.playlist_tracks('37i9dQZF1DX4JAvHpjipBk', fields='items').values():
            for x in val:
                for key, res in x.items():
                    if(key == 'track'):
                        for x, vals in res.items():
                            if x == 'id':
                                if vals not in trackids:
                                    trackids.append(vals)
    except:
        print('error')
        print(len(trackids))
        print(len(tracks))
        
        
    chart = billboard.ChartData('hot-100')
    tracks = []

    for track in trackids:
        Week = dt.datetime.now().strftime('%m/%d/%Y')
        Song = None
        Peak = None
        Performer = []
        SongID = None
        WeeksOnChart = None
        spotify_track_explicit = None
        spotify_track_duration_ms = None
        spotify_track_popularity = None
        danceability = None
        energy = None
        key = None
        loudness = None
        mode = None
        speechiness = None
        acousticness = None
        instrumentalness = None
        liveness = None
        valence = None
        tempo = None
        time_signature = None
        debut = None
        for k, v in sp.track(track).items():
            if k == 'artists':
                for artist in v:
                    for key, val in artist.items():
                        if key == 'name':
                            Performer.append(val)
            if k == 'duration_ms':
                spotify_track_duration_ms = v
            if k == 'explicit':
                spotify_track_explicit = v
            if k == 'popularity':
                spotify_track_popularity = v
            if k == 'name':
                Song = re.split("(\(feat.*\))", v)[0].strip()
        if sp.audio_features(track)[0]:
            for k, v in sp.audio_features(track)[0].items():
                if k == 'danceability':
                    danceability = v
                if k == 'energy':
                    energy = v
                if k == 'key':
                    key = v
                if k == 'loudness':
                    loudness = v
                if k == 'mode':
                    mode = v
                if k == 'speechiness':
                    speechiness = v
                if k == 'acousticness':
                    acousticness = v
                if k == 'instrumentalness':
                    instrumentalness = v
                if k == 'liveness':
                    liveness = v
                if k == 'valence':
                    valence = v
                if k == 'tempo':
                    tempo = v
                if k == 'time_signature':
                    time_signature = v
        for song in chart:
            if (Song == song.title and Performer[0] in song.artist):
                Performer[0] = song.artist
                if song.peakPos:
                    if song.rank == song.peakPos:
                        Peak = song.rank
                    else:
                        Peak = 'N/A'
                else:
                    Peak = song.rank
                if song.lastPos:
                    if (song.lastPos != song.peakPos and song.lastPos > song.rank):
                        debut = song.lastPos
                    else:
                        debut = 'N/A'
                else:
                    debut = song.rank
                WeeksOnChart = song.weeks

        if debut != 'N/A' and Peak != 'N/A':
            SongID = Song+Performer[0]
            tracks.append([Week, SongID, Song, Performer[0], Peak, WeeksOnChart, spotify_track_explicit, spotify_track_duration_ms, spotify_track_popularity, \
                          danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, debut])

    newtracks = pd.DataFrame(tracks, columns =['Week', 'SongID', 'Song', 'Performer', 'Peak', 'WeeksOnChart', 'spotify_track_explicit', 'spotify_track_duration_ms', 'spotify_track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'debut']) 

    newtracks["Lyrics"] = 0
    
    genius = lyricsgenius.Genius("lRSmSx2mp4ZU4ZovVyII_bDmIN2ROTR_0yU9HDCZI47VKNkZw-sBf5b4pruTYMam")
    genius.remove_section_headers = True
    genius.verbose = False
    genius.skip_non_songs = True
    
    for idx, values in newtracks.iterrows():
        try:
            lyrics = genius.search_song(values['Song'],values['Performer']).lyrics
            newtracks["Lyrics"][idx] = lyrics
        except:
            print("An exception occurred")
            
    newtracks.drop(newtracks[newtracks['Lyrics'] == 0].index, inplace = True)
    newtracks["LyricPositive"] = 0
    newtracks["LyricNeutral"] = 0
    newtracks["LyricNegative"] = 0
    
    sia = SentimentIntensityAnalyzer()
    for idx, values in newtracks.iterrows():
        if detect(values['Lyrics']) == 'en':
            num_positive = 0
            num_neutral = 0
            num_negative = 0
            for line in values['Lyrics'].splitlines():
                if line != "":
                    comp = sia.polarity_scores(line)   
                    comp = comp['compound']
                    if comp >= 0.5:
                        num_positive += 1
                    elif comp > -0.5 and comp < 0.5:
                        num_neutral += 1
                    else:
                        num_negative += 1
            num_total = num_negative + num_neutral + num_positive
            percent_negative = (num_negative/float(num_total))*100
            percent_neutral = (num_neutral/float(num_total))*100
            percent_positive = (num_positive/float(num_total))*100
            newtracks['LyricPositive'][idx] = percent_positive
            newtracks['LyricNeutral'][idx] = percent_neutral
            newtracks['LyricNegative'][idx] = percent_negative
            
    newtracks['LyricSentimentTotal'] = newtracks.apply(lambda row: row.LyricPositive + row.LyricNeutral + row.LyricNegative, axis=1)
    newtracks.drop(newtracks[newtracks['LyricSentimentTotal'] == 0].index, inplace = True)
    newtracks = newtracks.drop(columns=['LyricSentimentTotal'])
    
    # Creates New Twitter Dataset with Sentiment Fields
    twitter = newtracks[['Week', 'Song', 'Performer', 'SongID']]
    twitter['Tweets'] = 0
    twitter['TweetPositive'] = 0
    twitter['TweetNeutral'] = 0
    twitter['TweetNegative'] = 0
    
    # Twitter Data Collection
    iteration = 0
    for idx, values in twitter.iterrows():
        if idx > 32:
            if iteration == 15:
                time.sleep(500) # Makes the operation sleep for 8 minutes as to not overload API with a large number of requests
                iteration = 0
            tweets = []
            d = dt.timedelta(days=14)
            enddate = dt.datetime.strptime(values['Week'], "%m/%d/%y")
            startdate = enddate-d
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(values['Song'] + " " + values['Performer']).setSince(startdate.strftime("%Y-%m-%d")).setUntil(enddate.strftime("%Y-%m-%d")).setMaxTweets(100)
            tweet = got.manager.TweetManager.getTweets(tweetCriteria)
            stopwords12 = set(stopwords.words('english') + list(punctuation))
            for x in tweet:
                x = x.text
                x = x.lower() # convert text to lower-case
                x = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', x) # remove URLs
                x = re.sub('@[^\s]+', '', x) # remove usernames
                x = re.sub(r'#([^\s]+)', r'\1', x) # remove the # in #hashtag
                try:
                    # Checks if the tweets are in english for sentiment analysis
                    if detect(x) == 'en':
                        tweets.append(x)
                except:
                    print("LangDetect Error")
            twitter['Tweets'][idx] = tweets
            iteration+=1
            
    sia = SentimentIntensityAnalyzer()
    for idx, values in twitter.iterrows():
        newtweets = []
        song = values['Song'][0].lower()
        performer = values['Performer'][0].lower()
        for x in values['Tweets']:
            text = x.replace(song, "").replace(performer, "").replace('-','').strip()
            if text:
                newtweets.append(text)
        twitter['Tweets'][idx] = newtweets
        num_positive = 0
        num_neutral = 0
        num_negative = 0
        for x in newtweets:
            if x != "":
                comp = sia.polarity_scores(x)   
                comp = comp['compound']
                if comp >= 0.5:
                    num_positive += 1
                elif comp > -0.5 and comp < 0.5:
                    num_neutral += 1
                else:
                    num_negative += 1
        num_total = num_negative + num_neutral + num_positive
        if num_total != 0:
            percent_negative = (num_negative/float(num_total))*100
            percent_neutral = (num_neutral/float(num_total))*100
            percent_positive = (num_positive/float(num_total))*100
            twitter['TweetPositive'][idx] = percent_positive
            twitter['TweetNeutral'][idx] = percent_neutral
            twitter['TweetNegative'][idx] = percent_negative
            
    twitter['TweetSentimentTotal'] = twitter.apply(lambda row: row.TweetPositive + row.TweetNeutral + row.TweetNegative, axis=1)
    
    newtracks = pd.merge(newtracks, twitter, how='inner', on=['SongID'])
    
    newtracks = newtracks.drop(columns=['Week_y', 'Song_y', 'Performer_y'])
    newtracks = newtracks.rename(columns={"Week_x": "Week", "Song_x": "Song", "Performer_x": "Performer"})
    newtracks.drop(newtracks[newtracks['TweetSentimentTotal'] == 0].index, inplace = True)
    newtracks = newtracks.drop(columns=['TweetSentimentTotal'])
    
    newtracks = newtracks.reset_index()
    newtracks = newtracks.drop(columns=['index'])

    topics = []
    fs = gcsfs.GCSFileSystem(project='twitterbot-277618')
    with fs.open('datafortwitter/LDA.csv') as f:
        topics = pd.read_csv(f)
    
    def lemmatize_stemming(text):
        stemmer = PorterStemmer()
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
    words = []
    for lyrics in topics['Lyrics']:
        for word in gensim.utils.simple_preprocess(lyrics):
            if word not in gensim.parsing.preprocessing.STOPWORDS and len(word) > 3:
                words.append(lemmatize_stemming(word))
    add_stop_words = [word for word, count in Counter(words).most_common() if count > 750]
    
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                token = lemmatize_stemming(token)
                if token not in add_stop_words:
                    result.append(token)
        return result
    
    processed_docs = topics['Lyrics'].map(preprocess)
    
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
    newtracks['Topic 0'] = 0.0
    newtracks['Topic 1'] = 0.0
    newtracks['Topic 2'] = 0.0
    newtracks['Topic 3'] = 0.0
    newtracks['Topic 4'] = 0.0
    newtracks['Topic 5'] = 0.0
    newtracks['Topic 6'] = 0.0
    newtracks['Topic 7'] = 0.0
    newtracks['Topic 8'] = 0.0
    newtracks['Topic 9'] = 0.0
    
    for idx, values in newtracks.iterrows():
        bow_vector = dictionary.doc2bow(preprocess(values['Lyrics']))
        for indx, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
            newtracks['Topic ' + str(indx)][idx] = score
            
    featureset = newtracks
    featureset = featureset.drop(columns=['Week', 'SongID', 'Song','Performer', 'Tweets', 'Lyrics'])
    featureset['Classification'] = -1
    
    for idx, values in featureset.iterrows():
        number = -1
        if values['Peak'] == 1:
            number = 0
        elif values['Peak'] < 6:
            number = 1
        elif values['Peak'] < 16:
            number = 2
        elif values['Peak'] < 31:
            number = 3
        elif values['Peak'] < 51:
            number = 4
        elif values['Peak'] < 76:
            number = 5
        elif values['Peak'] < 101:
            number = 6
        else:
            number = -1
        featureset['Classification'][idx] = number
        
    featureset = featureset.drop(columns=['Peak'])
    
    for idx, values in featureset.iterrows():
        if isinstance(values['key'], int) == False:
            featureset['key'][idx] = -1
            
    featureset2 = []
    with fs.open('datafortwitter/FeaturesLimited.csv') as file:
        featureset2 = pd.read_csv(file)
    featureset2 = featureset2.drop(columns=['Unnamed: 0'])
    
    df=spark.createDataFrame(featureset2)
    testdf = spark.createDataFrame(featureset)
    
    RANDOM_SEED = 13579
    transformed_df = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))
    transformed_testdf = testdf.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

    # Training the Classifier
    rf = RandomForest.trainClassifier(transformed_df, numClasses=7, categoricalFeaturesInfo={}, \
        numTrees=25, featureSubsetStrategy="auto", impurity="gini", \
        maxDepth=10, maxBins= 32, seed=RANDOM_SEED)

    predictions = rf.predict(transformed_testdf.map(lambda x: x.features))
    labels_and_predictions = transformed_testdf.map(lambda x: x.label).zip(predictions)
    
    index = 0
    hitsongs = []
    predictionlist = predictions.collect()
    for x in predictionlist:
        if x == 1.0:
            hitsongs.append(index)
        index+=1
        
    update = "One of these songs will rank in the Top 5 Songs of Next Week's Billboard Hot 100 Chart:\n\n"
    sort = []
    def takeSecond(elem):
        return elem[2]
    for idx, values in newtracks.iterrows():
        if idx in hitsongs:
            sort.append([values['Song'], values['Performer'], values['spotify_track_popularity']])
    sort = sorted(sort, key=takeSecond, reverse=True)
    num = 0
    for x in sort:
        if num < 5:
            update = update + x[0]+' by '+x[1]+'\n'
            num+=1
            
    auth = tweepy.OAuthHandler('TvrYJIPHns4OPJCetpYg2RJw4', 'juNWwVL0l527PMTMCni6NQaFB5e5Ksjf3DXBreSkGCjdTEjzYF')
    auth.set_access_token('121839199-fJpZUrP3IS4R4MvoW8kaGg3PevusBjTSXu56mbe4', '5kAx7woytQWNRWuxuwgGAKX61onlluULLgPqHSXnfx6bW')

    api = tweepy.API(auth)
    api.update_status(update)
    time.sleep(604800)
}
    
    
       