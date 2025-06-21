import praw
import os
from dotenv import load_dotenv

load_dotenv()

class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.environ['REDDIT_CLIENT_ID'],
            client_secret=os.environ['REDDIT_CLIENT_SECRET'],
            user_agent=os.environ['USER_AGENT']
        )

    def get_subreddit(self, name):
        return self.reddit.subreddit(name)
