import json
from collections import defaultdict
from reddit_client import RedditClient
import time
from datetime import datetime, timedelta, UTC

class DataCollector:
    def __init__(self, subreddits=None):
        self.client = RedditClient()
        self.subreddits = ["news", "conservative", "conspiracy", "politics",
                            "worldnews", "MensRights", "The_Donald", "JordanPeterson",
                        "KotakuInAction", "TumblrInAction", "MGTOW", "SocialJusticeInAction"]

        # Will hold per‐user [(timestamp, subreddit)] lists
        self.user_data = defaultdict(list)

        # Will hold per‐subreddit { user, user, … }
        self.subreddit_users = defaultdict(set)

    def collect_user_posts(self, limit=5000, time_window="year"):
        time_filters = {
            "hour": 1,
            "day": 24,
            "week": 168,
            "month": 720,
            "year": 8640
        }
        time_threshold = datetime.now(UTC) - timedelta(hours=time_filters.get(time_window, 720))
        active_subreddits = []

        for sub in self.subreddits:
            try:
                subreddit = self.client.get_subreddit(sub)
                # Verify subreddit is accessible
                _ = subreddit.title
                active_subreddits.append(sub)
            except Exception as e:
                print(f"  → Skipping r/{sub}: {str(e)}")
                continue

        # Update to only active subreddits
        self.subreddits = active_subreddits
        print(f"Active subreddits: {', '.join(active_subreddits)}")

        for sub in self.subreddits:
            subreddit = self.client.get_subreddit(sub)
            print(f"Collecting from r/{sub}...")
            try:
                # Use .new(...) for temporal ordering
                for post in subreddit.new(limit=limit):
                    post_time = datetime.fromtimestamp(post.created_utc, UTC)
                    if post_time < time_threshold:
                        break
                    elif post.author:
                        username = post.author.name
                        timestamp = post.created_utc
                        text = post.title + " " + post.selftext
                        self.user_data[username].append((timestamp, sub, text))
                        self.subreddit_users[sub].add(username)
            except Exception as e:
                print(f"  → Failed to collect from r/{sub}: {e}")

    # In save_user_trajectory method
    def save_user_trajectory(self, filename="user_trajectory.json"):
        """Store [timestamp, subreddit, text] for each post"""
        serializable = {
            user: [[ts, sub, text] for ts, sub, text in posts]  # Changed to include text
            for user, posts in self.user_data.items()
        }
        with open(filename, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved user trajectory data to {filename}.")

    def save_subreddit_users(self, filename="subreddit_users.json"):
        serializable = {
            sub: list(users)
            for sub, users in self.subreddit_users.items()
        }
        with open(filename, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved subreddit→users data to {filename}.")
