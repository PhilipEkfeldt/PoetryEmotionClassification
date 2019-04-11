##################################################################################
# Project: Classifying Emotion in Poetry using Song Lyrics and Dual Attention Transfer Networks
# Script name: reddit_scraper_v0.1.py
# Description: This script scrapes data from the OCPoetry subreddit, and extracts the top 700 poems.
# Author: Yash Deshpande  
# Last modified: April 4th, 2019
# Comments/updates:
#
##################################################################################

# Required libraries
import pandas as pd 
import praw as pr
import datetime

# Initialization
start_time = datetime.datetime.now()
print("Initializing.")
subreddit_name = "OCPoetry"
reddit = pr.Reddit(client_id='qdDjSWUhfM9iYQ', client_secret='q0EwWDtIgKVCT9NjRF6dErBRDq0', user_agent='Reddit WebScraping')

required_flairs = ['Feedback Request', 'Feedback Received!']
top_posts = reddit.subreddit(subreddit_name).top(time_filter='all', limit=None)
posts = []
accepted_post_counter = 0

print("Extracting post data.")
for post_to_check in top_posts:

    if accepted_post_counter < 701:
        
        if str(post_to_check.link_flair_text) in required_flairs:
            posts.append([post_to_check.title, str(post_to_check.selftext)])
            accepted_post_counter +=1
    if accepted_post_counter % 50 == 0:
        print("Current accepted post count: "+str(accepted_post_counter))

posts = pd.DataFrame(posts, columns = ['title', 'text'])
posts['text'] = posts['text'].apply(str)
posts.to_csv("top_700_OCPoetry_posts.csv", sep='|', compression = None, encoding = 'utf-8')
pd.DataFrame(posts['text']).to_csv('MTurk_input.csv', index = False)
print("Script execution finished. Total time taken: "+str(datetime.datetime.now() - start_time))