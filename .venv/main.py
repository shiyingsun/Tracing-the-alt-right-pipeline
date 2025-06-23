from data_collector import DataCollector
from graph_builder import GraphBuilder
import networkx as nx
import time
from datetime import datetime, timedelta, UTC

def main():
    config = {
        "post_limit": 10000,
        "time_window": "year",
        "min_overlap": 3,
        "min_transition": 2,
        "transition_days": 14,
        "core_subs": ["news", "politics"],
        "radical_keywords": {
            "great_replacement": ["replacement", "white genocide"],
            "incel": ["blackpill", "chad", "stacy"],
            "qanon": ["q drop", "wwg1wga", "the storm"]
        },

        "subreddits": [
            "news", "conservative", "conspiracy", "politics",
            "worldnews", "MensRights", "JordanPeterson",
            "KotakuInAction", "SocialJusticeInAction"
        ]
    }
    collector = DataCollector(subreddits=config["subreddits"])

    collector = DataCollector()
    collector.collect_user_posts(
        limit=config["post_limit"],
        time_window=config["time_window"]
    )

    collector.save_user_trajectory("user_trajectory.json")
    collector.save_subreddit_users("subreddit_users.json")

    builder = GraphBuilder(
        user_trajectory_file="user_trajectory.json",
        subreddit_users_file="subreddit_users.json"
    )
    builder.load_user_data()

    print("\nRunning sentiment analysis...")
    start_time = time.time()
    sentiment_results = builder.analyze_sentiment()
    print(f"Sentiment analysis completed in {time.time() - start_time:.2f}s")

    co_graph = builder.build_coposting_graph(min_overlap=2)
    traj_graph = builder.build_trajectory_graph(
        min_transition=2,
        time_window=14
    )

    print("\nRunning sentiment analysis...")
    start_time = time.time()
    sentiment_results = builder.analyze_sentiment()
    print(f"Sentiment analysis completed in {time.time() - start_time:.2f}s")

    co_graph = builder.build_coposting_graph(min_overlap=3)
    traj_graph = builder.build_trajectory_graph(
        min_transition=2,
        time_window=14
    )

    print("\nCalculating radiality scores...")
    radial_scores = builder.calculate_radiality_score(
        co_graph,
        core_subs=config["core_subs"]
    )
    print("Radiality Scores (distance from core):")
    for sub, score in radial_scores.items():
        print(f"  r/{sub}: {score}")

    print("\nAnalyzing meme propagation...")
    meme_results = builder.track_meme_propagation(
        keywords=config["radical_keywords"]
    )
    print("Meme Propagation Summary:")
    for meme, data in meme_results.items():
        print(f"  {meme}: first appeared in r/{data['subreddit']} at {datetime.utcfromtimestamp(data['timestamp'])}")

    print("\nRunning temporal analysis...")
    time_windows = [
        (datetime.utcnow() - timedelta(days=90), datetime.utcnow()),  # Last 3 months
        (datetime.utcnow() - timedelta(days=365), datetime.utcnow() - timedelta(days=91))  # Previous 9 months
    ]
    temporal_graphs = builder.build_temporal_graphs(time_windows)

    for i, (start, end) in enumerate(time_windows):
        print(f"\nTime Window {i + 1}: {start.date()} to {end.date()}")
        graph = temporal_graphs[i]
        pipeline_strength, avg_time = builder.calculate_pipeline_strength(graph)
        print(f"Pipeline strength: {pipeline_strength} users")
        if avg_time:
            print(f"Average time to radicalization: {avg_time:.1f} days")

        nx.write_gexf(graph, f"temporal_graph_window_{i + 1}.gexf")

    builder.draw_coposting_graph(co_graph)
    builder.draw_trajectory_graph(traj_graph)
    builder.draw_combined_graph(co_graph, traj_graph)

    builder.detect_communities(co_graph)
    builder.analyze_centrality(traj_graph)

    nx.write_gexf(co_graph, "coposting_graph.gexf")
    nx.write_gexf(traj_graph, "trajectory_graph.gexf")

if __name__ == "__main__":
    main()

# import json
# from graph_builder import GraphBuilder
# import networkx as nx
# import time
# from datetime import datetime, timedelta, UTC
#
# def main():
#     config = {
#                 "post_limit": 5000,
#                 "time_window": "year",
#                 "min_overlap": 3,
#                 "min_transition": 2,
#                 "transition_days": 14,
#                 "core_subs": ["news", "politics"],
#                 "radical_keywords": {
#                     "great_replacement": ["replacement", "white genocide"],
#                     "incel": ["blackpill", "chad", "stacy"],
#                     "qanon": ["q drop", "wwg1wga", "the storm"]
#                 },
#                 # Only include accessible subreddits
#                 "subreddits": [
#                     "news", "conservative", "conspiracy", "politics",
#                     "worldnews", "MensRights", "JordanPeterson",
#                     "KotakuInAction", "SocialJusticeInAction"
#                 ]
#             }
#
#     builder = GraphBuilder(
#         user_trajectory_file="user_trajectory.json",
#         subreddit_users_file="subreddit_users.json"
#     )
#     builder.load_user_data()
#
#     print("\nRunning sentiment analysis...")
#     start_time = time.time()
#     sentiment_results = builder.analyze_sentiment()
#     print(f"Sentiment analysis completed in {time.time() - start_time:.2f}s")
#
#     co_graph = builder.build_coposting_graph(min_overlap=config["min_overlap"])
#     traj_graph = builder.build_trajectory_graph(
#         min_transition=config["min_transition"],
#         time_window=config["transition_days"]
#     )
#
#     print("\nCalculating radiality scores...")
#     radial_scores = builder.calculate_radiality_score(
#         co_graph,
#         core_subs=config["core_subs"]
#     )
#     print("Radiality Scores (distance from core):")
#     for sub, score in radial_scores.items():
#         print(f"  r/{sub}: {score}")
#
#     print("\nAnalyzing meme propagation...")
#     meme_results = builder.track_meme_propagation(
#         keywords=config["radical_keywords"]
#     )
#     print("Meme Propagation Summary:")
#     for meme, data in meme_results.items():
#         print(f"  {meme}: first appeared in r/{data['subreddit']} at {datetime.utcfromtimestamp(data['timestamp'])}")
#
#     print("\nRunning temporal analysis...")
#     time_windows = [
#         (datetime.now(UTC) - timedelta(days=90), datetime.now(UTC)),
#         (datetime.now(UTC) - timedelta(days=365), datetime.now(UTC) - timedelta(days=91))
#     ]
#     temporal_graphs = builder.build_temporal_graphs(time_windows)
#
#     for i, (start, end) in enumerate(time_windows):
#         print(f"\nTime Window {i + 1}: {start.date()} to {end.date()}")
#         graph = temporal_graphs[i]
#         pipeline_strength, avg_time = builder.calculate_pipeline_strength(graph)
#         print(f"Pipeline strength: {pipeline_strength} users")
#         if avg_time:
#             print(f"Average time to radicalization: {avg_time:.1f} days")
#         nx.write_gexf(graph, f"temporal_graph_window_{i + 1}.gexf")
#
#     builder.draw_coposting_graph(co_graph)
#     builder.draw_trajectory_graph(traj_graph)
#     builder.draw_combined_graph(co_graph, traj_graph)
#
#     builder.detect_communities(co_graph)
#     builder.analyze_centrality(traj_graph)
#
#     nx.write_gexf(co_graph, "coposting_graph.gexf")
#     nx.write_gexf(traj_graph, "trajectory_graph.gexf")
#
# if __name__ == "__main__":
#     main()