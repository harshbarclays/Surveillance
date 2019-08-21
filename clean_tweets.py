import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords')
import pandas as pd


def clean_all_tweets_apply_model(df, curr_1, curr_2, ccypair):
    # Identifying retweets
    df['is_retweet'] = df['Message'].apply(lambda x: x[:2] == 'RT')
    df['is_retweet'].sum()

    # Most repeated tweets, top 10
    res =  df.groupby(['Message']).size().reset_index(name='Freq').sort_values('Freq', ascending=False).head(10)
    res.reset_index(drop=True, inplace=True)

    def retweets(tweet):
        return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

    def mentions(tweet):
        return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

    def hashtags(tweet):
        return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

    # make new columns for retweeted usernames, mentioned usernames and hashtags
    df['retweeted'] = df.Message.apply(retweets)
    df['mentioned'] = df.Message.apply(mentions)
    df['hashtags'] = df.Message.apply(hashtags)

    # take the rows from the hashtag columns where there are actually hashtags
    hashtags_list_df = df.loc[
        df.hashtags.apply(
            lambda hashtags_list: hashtags_list != []
        ), ['hashtags']]


    def remove_links(tweet):
        '''Takes a string and removes web links from it'''
        tweet = re.sub(r'http\S+', '', tweet)  # remove http links
        tweet = re.sub(r'bit.ly/\S+', '', tweet)  # rempve bitly links
        tweet = tweet.strip('[link]')  # remove [links]
        return tweet

    def remove_users(tweet):
        '''Takes a string and removes retweet and @user information'''
        tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove retweet
        tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove tweeted at
        return tweet

    my_stopwords = nltk.corpus.stopwords.words('english')
    word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

    # cleaning master function
    def clean_tweet(tweet, bigrams=False):
        tweet = remove_users(tweet)
        tweet = remove_links(tweet)
        tweet = tweet.lower()  # lower case
        tweet = re.sub('[' + my_punctuation + ']+', ' ', tweet)  # strip punctuation
        tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
        tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
        tweet = re.sub(curr_1, '', tweet)  # remove numbers
        tweet = re.sub(curr_2, '', tweet)  # remove numbers
        tweet = re.sub(ccypair, '', tweet)  # remove numbers

        tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords]  # remove stopwords

        #     tweet_token_list = [word_rooter(word) if '#' not in word else word
        #                         for word in tweet_token_list] # apply word rooter
        if bigrams:
            tweet_token_list = tweet_token_list + [tweet_token_list[i] + '_' + tweet_token_list[i + 1]
                                                   for i in range(len(tweet_token_list) - 1)]
        tweet = ' '.join(tweet_token_list)
        return tweet

    df['clean_tweet'] = df.Message.apply(clean_tweet)

    # Applying Topic Modeling
    # the vectorizer object will be used to transform text to vector form
    vectorizer = CountVectorizer(token_pattern='\w+|\$[\d\.]+|\S+')

    # apply transformation
    tf = vectorizer.fit_transform(df['clean_tweet']).toarray()

    # tf_feature_names tells us what word each column in the matric represents
    tf_feature_names = vectorizer.get_feature_names()

    number_of_topics = 5

    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
    model.fit(tf)

    def display_topics(model, feature_names, no_top_words):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)] = ['{}'.format(feature_names[i])
                                                          for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)] = ['{:.1f}'.format(topic[i])
                                                            for i in topic.argsort()[:-no_top_words - 1:-1]]
        return pd.DataFrame(topic_dict)

    no_top_words = 10
    lda_res = display_topics(model, tf_feature_names, no_top_words)

    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", model.score(tf))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", model.perplexity(tf))

    log_likelihodd = model.score(tf)
    perplexity = model.perplexity(tf)

    return res, hashtags_list_df, lda_res, log_likelihodd, perplexity