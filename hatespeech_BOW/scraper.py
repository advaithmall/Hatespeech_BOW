import snscrape.modules.twitter as sntwitter
import numpy as np
from IPython.display import display
import pandas as pd
import pprint

#Created a list to append all tweet attributes(data)
print("Please enter a valid twitter username: ")
user_name = input("")
print("Please enter number of tweets: ")
n = input()
attributes_container = []
text = ""
#Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'from:{user_name}').get_items()):
    #print("i = -> ",i)
    if i > 1000:
        break
    attributes_container.append(
       [tweet.content])
    
#Creating a dataframe from the tweets list above
tweets_df = pd.DataFrame(attributes_container, columns=[
   "tweet"])
display(tweets_df.to_string())
tweets_df.to_csv('file.csv')
