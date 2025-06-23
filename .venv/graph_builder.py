import json
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict, Counter
from textblob import TextBlob
import time
import math, numpy as np

class GraphBuilder:
    def __init__(self,
                 user_trajectory_file="user_trajectory.json",
                 subreddit_users_file=None,
                 subreddits=None):
        self.user_trajectory_file = user_trajectory_file
        self.subreddit_users_file = subreddit_users_file
        self.subreddits = subreddits or ["news", "conservative", "conspiracy", "politics",
                            "worldnews", "MensRights", "The_Donald", "JordanPeterson",
                        "KotakuInAction", "TumblrInAction", "MGTOW", "SocialJusticeInAction"]

        self.user_data = {}
        self.subreddit_users = defaultdict(set)
        self.subreddit_posts = defaultdict(list)

    def load_user_data(self):
        with open(self.user_trajectory_file, "r") as f:
            raw_data = json.load(f)
            self.user_data = {
                user: [(item[0], item[1], item[2]) for item in posts]
                for user, posts in raw_data.items()
            }

    def load_subreddit_users(self):
        if not self.subreddit_users_file:
            raise ValueError("No subreddit_users_file provided.")
        with open(self.subreddit_users_file, "r") as f:
            raw = json.load(f)
        for sub, users in raw.items():
            if sub in self.subreddits:
                self.subreddit_users[sub] = set(users)

    def build_coposting_graph(self, min_overlap=3):
        if not self.subreddit_users and self.user_data:
            for user, posts in self.user_data.items():
                for ts, sub, _ in posts:
                    if sub in self.subreddits:
                        self.subreddit_users[sub].add(user)

        G = nx.Graph()
        for sub in self.subreddits:
            G.add_node(sub)

        for i, sub1 in enumerate(self.subreddits):
            for sub2 in self.subreddits[i+1:]:
                shared = self.subreddit_users[sub1].intersection(self.subreddit_users[sub2])
                if len(shared) > min_overlap:
                    G.add_edge(sub1, sub2, weight=len(shared))

        return G

    def draw_coposting_graph(self, G):
        deg = dict(G.degree())
        btw = nx.betweenness_centrality(G, weight='weight')

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        except:
            pos = nx.spring_layout(G, k=2.0, iterations=300, seed=42)

        strength = {
            n: sum(d['weight'] for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
        min_s, max_s = min(strength.values()), max(strength.values())

        r_min, r_max = 1.0, 1.3

        new_pos = {}
        for n, (x, y) in pos.items():
            θ = math.atan2(y, x)
            norm = (strength[n] - min_s) / (max_s - min_s) if max_s > min_s else 0
            r = r_min + (1 - norm) * (r_max - r_min)
            new_pos[n] = np.array([math.cos(θ) * r, math.sin(θ) * r])
        pos = new_pos

        fig, ax = plt.subplots(figsize=(16, 12))

        raw_w = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(raw_w) if raw_w else 1

        for (u, v), w in zip(G.edges(), raw_w):
            width = (w / max_w) * 5 + 0.5
            arrow = FancyArrowPatch(
                pos[u], pos[v],
                arrowstyle='-',
                connectionstyle="arc3,rad=0.2",
                linewidth=width,
                color='red',
                alpha=0.5,
                zorder=1
            )
            ax.add_patch(arrow)

        node_sizes = [300 + deg[n] * 200 for n in G.nodes()]
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=[btw.get(n, 0) for n in G.nodes()],
            cmap=plt.cm.OrRd,
            node_size=node_sizes,
            edgecolors='black',
            linewidths=1.5,
            ax=ax
        )
        nodes.set_zorder(2)

        for node, (x, y) in pos.items():
            ax.text(
                x, y, node,
                fontsize=10, fontweight='bold',
                ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=4, foreground='white')],
                zorder=3
            )

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.OrRd,
            norm=plt.Normalize(vmin=min(btw.values(), default=0),
                               vmax=max(btw.values(), default=1))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Betweenness Centrality', fontsize=12)

        ax.set_title('Subreddit Co-posting Graph (Radialized)', fontsize=16)
        ax.axis('off')
        plt.tight_layout(pad=2)
        plt.show()

        edges_sorted = sorted(G.edges(data=True),
                              key=lambda x: x[2]['weight'],
                              reverse=True)
        print("Top 3 co-posting overlaps:")
        for u, v, d in edges_sorted[:3]:
            print(f"  {u} – {v}: {d['weight']} shared users")

    def build_trajectory_graph(self, min_transition=2, time_window=30, data_filter=None):
        transitions = Counter()

        for user, posts in self.user_data.items():
            post_meta = [(ts, sub) for ts, sub, text in posts]
            sorted_posts = sorted(posts, key=lambda x: x[0])

            sequence = []
            last_sub = None
            for ts, sub, _ in sorted_posts:
                if sub != last_sub:
                    sequence.append((ts, sub))
                    last_sub = sub

            for i in range(len(sequence) - 1):
                src_time, src = sequence[i]
                dst_time, dst = sequence[i + 1]
                time_diff = dst_time - src_time

                max_seconds = time_window * 86400
                if (time_diff <= max_seconds and
                        src in self.subreddits and
                        dst in self.subreddits):
                    transitions[(src, dst)] += 1

        G = nx.DiGraph()
        for sub in self.subreddits:
            G.add_node(sub)

        for (src, dst), count in transitions.items():
            if count >= min_transition:
                G.add_edge(src, dst, weight=count)

        return G

    def draw_trajectory_graph(self, G):
        for u, v, data in G.edges(data=True):
            data['distance'] = data['weight']

        try:
            pos = nx.kamada_kawai_layout(G, weight='distance')
        except:
            pos = nx.spring_layout(G, k=3.0, scale=10.0, iterations=300, seed=42)

        strength = {
            n: sum(d['weight'] for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
        min_s, max_s = min(strength.values()), max(strength.values())

        r_min, r_max = 1.0, 1.3

        rad_pos = {}
        for n, (x, y) in pos.items():
            θ = math.atan2(y, x)
            norm = (strength[n] - min_s) / (max_s - min_s) if max_s > min_s else 0
            r = r_min + (1 - norm) * (r_max - r_min)
            rad_pos[n] = np.array([math.cos(θ) * r, math.sin(θ) * r])
        pos = rad_pos

        fig, ax = plt.subplots(figsize=(14, 10))

        raw_w = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(raw_w) if raw_w else 1
        for (u, v), w in zip(G.edges(), raw_w):
            width = (w / max_w) * 6 + 0.5
            rad = 0.2
            patch = FancyArrowPatch(
                pos[u], pos[v],
                arrowstyle='-|>',
                mutation_scale=15,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=width,
                color='crimson',
                alpha=0.8,
                zorder=1
            )
            ax.add_patch(patch)

        deg = dict(G.degree())
        node_sizes = [800 + deg[n] * 200 for n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color='lightcoral',
            edgecolors='black',
            linewidths=1.2,
            ax=ax
        )

        for node, (x, y) in pos.items():
            ax.text(
                x, y, node,
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=4, foreground='white')],
                zorder=3
            )

        edge_labels = {
            (u, v): data['weight']
            for u, v, data in G.edges(data=True)
            if data['weight'] >= 10
        }

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=10,
            ax=ax
        )

        ax.set_title("User Trajectories Between Subreddits", fontsize=18)
        ax.axis('off')
        plt.tight_layout(pad=2)
        plt.show()

    def draw_combined_graph(self, G_undirected, G_directed):
        """
        Overlay co-posting (light gray) and trajectory (red arrows) with
        radialized layout so low-interaction nodes drift outward.
        """

        pos = nx.spring_layout(
            G_undirected,
            k=1.8,
            iterations=300,
            seed=42
        )

        strength = {}
        for n in G_undirected.nodes():
            w_u = sum(d['weight'] for _, _, d in G_undirected.edges(n, data=True))
            w_d = sum(d['weight'] for _, _, d in G_directed.in_edges(n, data=True)) \
                  + sum(d['weight'] for _, _, d in G_directed.out_edges(n, data=True))
            strength[n] = w_u + w_d

        min_s, max_s = min(strength.values()), max(strength.values())

        # use to adjust spread
        r_min, r_max = 1.0, 1.4

        new_pos = {}
        for n, (x, y) in pos.items():
            θ = math.atan2(y, x)
            norm = (strength[n] - min_s) / (max_s - min_s) if max_s > min_s else 0
            r = r_min + (1 - norm) * (r_max - r_min)
            new_pos[n] = np.array([math.cos(θ) * r, math.sin(θ) * r])
        pos = new_pos

        raw_w_u = [G_undirected[u][v]['weight'] for u, v in G_undirected.edges()]
        max_w_u = max(raw_w_u) if raw_w_u else 1
        widths_u = [(w / max_w_u) * 6 for w in raw_w_u]

        plt.figure(figsize=(20, 14))

        node_sizes = [
            len(self.subreddit_users.get(node, [])) * 12 + 200
            for node in G_undirected.nodes()
        ]
        nx.draw_networkx_nodes(
            G_undirected, pos,
            node_size=node_sizes,
            node_color="white",
            edgecolors="black",
            linewidths=1.5
        )

        nx.draw_networkx_edges(
            G_undirected, pos,
            width=widths_u,
            edge_color="lightgray",
            alpha=0.8
        )

        raw_w_d = [G_directed[u][v]['weight'] for u, v in G_directed.edges()]
        max_w_d = max(raw_w_d) if raw_w_d else 1
        for u, v, data in G_directed.edges(data=True):
            w = data['weight']
            nx.draw_networkx_edges(
                G_directed, pos,
                edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={0.1 if u != v else 0.3}",
                width=(w / max_w_d) * 20,
                edge_color="crimson",
                arrowstyle='->',
                arrowsize=25,
                alpha=1.0
            )

        for node, (x, y) in pos.items():
            plt.text(
                x, y, node,
                fontsize=12, fontweight="bold",
                ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=3, foreground="white")]
            )

        plt.axis("off")
        plt.gcf().set_facecolor('white')
        plt.title("Combined Graph (Radialized Layout)", fontsize=16)
        plt.connect('button_press_event', self._on_click)
        plt.tight_layout()
        plt.show()

        edges_sorted = sorted(
            G_undirected.edges(data=True),
            key=lambda x: x[2]['weight'],
            reverse=True
        )

        print("Top 3 co-posting overlaps:")
        for u, v, d in edges_sorted[:3]:
            print(f"  {u} – {v}: {d['weight']} shared users")

        transitions = sorted(
            [((u, v), data['weight']) for u, v, data in G_directed.edges(data=True)],
            key=lambda x: x[1],
            reverse=True
        )
        print("\nTop 3 subreddit transitions:")
        for (src, dst), cnt in transitions[:3]:
            print(f"  {src} → {dst}: {cnt} users")

    def detect_communities(self, G):
        """Add community detection to analysis using networkx built-in"""
        if not nx_comm:
            print("Warning: Community detection module not available")
            return G

        communities = nx_comm.louvain_communities(G.to_undirected(), seed=42)

        partition = {}
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                partition[node] = comm_id
        nx.set_node_attributes(G, partition, "community")

        comm_mapping = defaultdict(list)
        for node, comm_id in partition.items():
            comm_mapping[comm_id].append(node)

        print("\nDetected Communities:")
        for comm_id, nodes in comm_mapping.items():
            print(f"Community {comm_id}: {', '.join(nodes)}")

        return G

    def analyze_centrality(self, G):
        """Compute and print key centrality metrics"""
        print("\nCentrality Analysis:")
        print("Betweenness:")
        betweenness = nx.betweenness_centrality(G)
        for node, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {node}: {score:.4f}")

        print("\nEigenvector Centrality:")
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        for node, score in sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {node}: {score:.4f}")

    def _on_click(self, event):
        """Basic zoom functionality"""
        ax = event.inaxes
        if not ax: return

        if event.button == 1:  # Left click
            ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
            ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
        elif event.button == 3:  # Right click
            ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
            ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
        plt.draw()

    def analyze_sentiment(self):
        sentiment_results = {}
        for user, posts in self.user_data.items():
            user_sentiments = []
            for post in posts:
                text = ""  # Should be actual post text from data collection
                if text:
                    analysis = TextBlob(text)
                    polarity = analysis.sentiment.polarity
                    user_sentiments.append(polarity)
            if user_sentiments:
                sentiment_results[user] = np.mean(user_sentiments)
        return sentiment_results

    def calculate_radiality_score(self, G, core_subs=["news"]):
        scores = {}
        for node in G.nodes:
            try:
                paths = [nx.shortest_path_length(G, core, node)
                         for core in core_subs if nx.has_path(G, core, node)]
                scores[node] = min(paths) if paths else float('inf')
            except:
                scores[node] = float('inf')
        return scores

    def track_meme_propagation(self, keywords):
        meme_db = defaultdict(dict)

        for sub, posts in self.subreddit_posts.items():
            for post in posts:
                detected_memes = self.detect_memes(post['text'])
                for meme in detected_memes:
                    if meme not in meme_db or meme_db[meme]['timestamp'] > post['timestamp']:
                        meme_db[meme] = {'subreddit': sub, 'timestamp': post['timestamp']}

        return meme_db

    def build_temporal_graphs(self, time_windows):
        graphs = []
        for start, end in time_windows:
            start_ts = time.mktime(start.timetuple())
            end_ts = time.mktime(end.timetuple())

            graph = self.build_trajectory_graph(
                min_transition=2,
                time_window=30,
                data_filter=lambda x: start_ts <= x[0] <= end_ts
            )
            graphs.append(graph)
        return graphs

    def calculate_pipeline_strength(self, G):
        mainstream = ["news", "worldnews"]
        radical = ["conservative", "conspiracy"]

        pipeline_strength = 0
        total_time = 0
        for user, posts in self.user_data.items():
            timeline = sorted(posts, key=lambda x: x[0])
            mainstream_ts = None
            radical_ts = None

            for ts, sub, _ in timeline:
                if sub in mainstream and mainstream_ts is None:
                    mainstream_ts = ts
                if sub in radical:
                    radical_ts = ts if radical_ts is None else min(radical_ts, ts)

            if mainstream_ts and radical_ts and radical_ts > mainstream_ts:
                pipeline_strength += 1
                total_time += (radical_ts - mainstream_ts) / 86400  # Convert to days

        avg_time = total_time / pipeline_strength if pipeline_strength else None
        return pipeline_strength, avg_time