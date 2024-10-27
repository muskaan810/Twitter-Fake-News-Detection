import nest_asyncio
import asyncio
import pandas as pd
from twscrape import API, gather

nest_asyncio.apply()

async def fetch_tweets(keyword):
   df = pd.read_csv("Tweets_Scraped\\all_tweets_rf_final.csv")
   df = df.sort_values(by=["likes", "retweets"], ascending=False)
   return df
  