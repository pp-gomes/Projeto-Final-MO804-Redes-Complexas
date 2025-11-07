import praw  # Reddit API wrapper
import networkx as nx  # NetworkX for graph operations
from time import sleep  # Used for retry logic
from datetime import datetime
import pytz  # Timezone handling
from merge_graphs import GraphMerger  # Custom graph merging utility
from make_adj_list import Separator  # Custom adjacency list utility
from tqdm import tqdm  # Progress bar for loops

class RedditGraphExtractor:
    """
    Extracts user interaction graphs from Reddit comments using PRAW and NetworkX.
    """
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize Reddit API client and an empty directed multigraph.
        """
        self.reddit = praw.Reddit(
            client_id="Oc3-NZkgYZPJwsUlUNBufQ",
            client_secret="H0GCmuJ1gS46COMDQznna3j354I8bw",
            user_agent=u"meu_graph:v1.0 (by u/ppgomes)"
        )
        self.graph = nx.MultiDiGraph()
    
    def _dfs(self, parent, comment, subreddit_name, history):
        """
        Depth-first traversal of comment tree to build user interaction edges.
        Adds an edge from the current comment's author to the parent comment's author.
        Retries on failure after sleeping for 60 seconds.
        """
        try:
            current_author = comment.author
            parent_author = parent.author

            # Append current comment body to history string
            history += " |~:~| " + comment.body

            if current_author and parent_author:
                current_author = current_author.name
                parent_author = parent_author.name

                # Ignore AutoModerator comments
                if current_author != "AutoModerator" and parent_author != "AutoModerator":
                    self.graph.add_edge(
                        current_author, parent_author,
                        subreddit=subreddit_name,
                        comments=history,
                        score=comment.score,
                        submissionDate = str(datetime.fromtimestamp(comment.created_utc, tz=pytz.utc).strftime('%Y-%m-%d %H:%M:%S %Z%z')),
                        collectionDate = str(datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S %Z%z'))
                    )

            # Recursively process child comments
            for child in comment.replies:
                try:
                    self._dfs(comment, child, subreddit_name, history)
                except Exception as e:
                    sleep(60)
                    self._dfs(comment, child, subreddit_name, history)

        except Exception as e:
            sleep(60)
            self._dfs(parent, comment, subreddit_name, history)
    
    def extract_interactions(self, subreddit_name, post_limit, min_comments, max_comments, max_posts):
        """
        Extracts user interactions from a subreddit and builds a graph.
        Only processes posts with a number of comments within the specified range.
        Uses tqdm to show progress.
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        tracked_posts = 0
        pbar = tqdm(total=max_posts, desc=f"r/{subreddit_name}")

        for submission in subreddit.new(limit=post_limit):
            if tracked_posts >= max_posts:
                break
            try:
                num_comments = submission.num_comments

                # Skip posts outside the comment range
                if not (max_comments >= num_comments >= min_comments):
                    continue

                # Load all comments for the submission
                submission.comments.replace_more(limit=None)
                title = submission.title
                for top_level_comment in submission.comments:
                    try:
                        post = top_level_comment.body
                        # Traverse second-level comments
                        for second_level_comment in top_level_comment.replies:
                            try:
                                self._dfs(top_level_comment, second_level_comment, subreddit_name, title + " |~:~| " + post)
                            except Exception as e:
                                sleep(60)
                                self._dfs(top_level_comment, second_level_comment, subreddit_name, title + " |~:~| " + post)
                    except Exception as e:
                        sleep(60)

                tracked_posts += 1
                pbar.update(1)

            except Exception as e:
                sleep(60)

    def save_graph(self, path):
        """
        Save the extracted graph to a GEXF file.
        """
        nx.write_gexf(self.graph, path)

if __name__ == "__main__":
    # Main script for extracting Reddit graphs from a list of subreddits
    import argparse
    from dotenv import load_dotenv
    import os
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract Reddit interactions and build a graph.")
    parser.add_argument("--post_limit", type=int, default=100000, help="Maximum number of posts to process per subreddit.")
    parser.add_argument("--min_comments", type=int, default=200, help="Minimum number of comments required for a post to be considered.")
    parser.add_argument("--max_comments", type=int, default=2000, help="Maximum number of comments allowed for a post to be considered.")
    parser.add_argument("--max_posts", type=int, default=30, help="Maximum number of posts to track per subreddit.")
    parser.add_argument('--merge_graphs', action='store_true', help='Create a separate file with all graphs merged.')
    parser.add_argument('--dump_messages', action='store_true', help='Create a JSON file of a python dictionary storing all edges between any two nodes')

    args = parser.parse_args()

    load_dotenv()

    # Create output folder, avoid overwriting existing folders
    folder_name = "reddit-graph"
    folder_path = f"./{folder_name}/"

    counter = 1
    while os.path.exists(folder_path):
        folder_path = f"./{folder_name}({counter})/"
        counter += 1

    os.makedirs(folder_path, exist_ok=False)

    # Read subreddits from file and process each
    with open('subreddits.txt', 'r') as file:
        for subreddit in file:
            extractor = RedditGraphExtractor(
                client_id=os.getenv('REDDIT_CLIENT_ID'), client_secret=os.getenv('REDDIT_CLIENT_SECRET'), user_agent=os.getenv('USER_AGENT')
            )
            extractor.extract_interactions(
                subreddit.strip(), post_limit=args.post_limit, min_comments=args.min_comments, max_comments=args.max_comments, max_posts=args.max_posts
            )
            extractor.save_graph(os.path.join(folder_path, f"{subreddit.strip()}.gexf"))

    # Optionally merge graphs and/or dump messages
    if args.merge_graphs or args.dump_messages:
        MergedGraph = GraphMerger(folder_path)
        MergedGraph.merge()
        if args.merge_graphs:
            MergedGraph.save()
        if args.dump_messages:
            messages = Separator(MergedGraph.merged_graph)
            messages.separate()
            messages.dump(folder_path)
