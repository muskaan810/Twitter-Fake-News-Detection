import pandas as pd
from classify import preprocess_tweet
import nest_asyncio
import asyncio
from datetime import datetime, timedelta
from twscrape import API, gather
import time
import random
#from twscrape import fetch_tweets as twscrape_fetch_tweets

def scrape_classify_store_tweets(keyword, filename):
    '''try:
        
# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def fetch_tweets(api, keyword, since_date, until_date, max_tweets=5000):
    query = f"{keyword} since:{since_date} until:{until_date} lang:en"
    tweets = await gather(api.search(query, limit=max_tweets))
    data = [{
        "date": tweet.date,
        "id": tweet.id,
        "content": tweet.rawContent,
        "username": tweet.user.username,
        "lang": tweet.lang,
        "likes": tweet.likeCount,
        "retweets": tweet.retweetCount,
        "replies": tweet.replyCount,
        "hashtags": tweet.hashtags,
        "location": tweet.user.location,
        "followers_count": tweet.user.followersCount,
        "friends_count": tweet.user.friendsCount,
        "profile_image_url": tweet.user.profileImageUrl,
        "verified": tweet.user.verified
    } for tweet in tweets]

    df = pd.DataFrame(data)
    return df

async def main():
    # Initialize API
    api = API()

    # Add your accounts (replace with actual credentials)
    await api.pool.add_account("username", "pwd", "email@gmail.com", "pwd")

    # Log in to all accounts
    await api.pool.login_all()

    # Define the keyword and search parameters
    keyword = "@Tesla OR #Tesla"
    end_time = datetime.strptime("2024-07-05 23:59:59", "%Y-%m-%d %H:%M:%S")
    stop_time = datetime.strptime("2024-07-05 00:00:00", "%Y-%m-%d %H:%M:%S")
    output_file = 'C://Users//Mushaan Khubchandani//Downloads//ts07-05.csv'

    while end_time > stop_time:
        try:
            until_date = end_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
            since_date = "2024-07-05_00:00:00_UTC"

            df = await fetch_tweets(api, keyword, since_date, until_date, max_tweets=5000)

            if not df.empty:
                # Append the new data to the CSV file
                df.to_csv(output_file, mode='a', header=not pd.read_csv(output_file).empty)

                # Print the number of tweets fetched
                print(f"Fetched {len(df)} tweets from {since_date} to {until_date} at {datetime.now()}")
            else:
                print(f"No tweets found from {since_date} to {until_date}")

            # Update end_time for the next iteration
            end_time -= timedelta(hours=2)

            # Introduce a random delay between 15 and 20 minutes
            sleep_time = random.randint(900, 1200)  # Random time between 15 (900s) and 20 (1200s) minutes
            print(f"Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            time.sleep(60)  # Wait for a minute before retrying in case of an error

'''
if __name__ == "__main__":
    keyword = 'Tesla'  # Example keyword
    filename = 'Tweets_Scraped\\tesla-07-05.csv'  # Your existing CSV file path
    scrape_classify_store_tweets(keyword, filename)
