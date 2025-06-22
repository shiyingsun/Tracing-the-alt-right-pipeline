# Graph Theory and Radicalization on Reddit

This project investigates online radicalization patterns on Reddit using graph-theoretic methods. It collects user posting data from selected political and ideological subreddits, builds co-posting and trajectory networks, and applies centrality, community detection, and radiality analyses.

## Features

- **Data Collection**: Uses `praw` to fetch recent posts and build per-user trajectories.  
- **Co-Posting Graph**: Undirected, weighted by number of shared users.  
- **Trajectory Graph**: Directed, weighted by subreddit transitions within a time window.  
- **Combined Overlay**: Merges co-posting and trajectory edges in one layout.  
- **Network Analyses**:  
  - Betweenness and eigenvector centrality  
  - Community detection (Louvain)  
  - k-Shell decomposition & radiality scoring  
  - Temporal diffusion and cascade metrics  

## Installation

1. Clone this repository  
   ```bash
   git clone https://github.com/<your-username>/reddit-radicalization-graph.git
   cd reddit-radicalization-graph
2. Create and activate a virtual enviroment
   ```bash
   python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
3. Install dependencies
   ```bash
   pip install -r requirements.txt
4. Create a .env file with your Reddit API credentials
   ```env
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   USER_AGENT=your_user_agent
## Usage
1. Collect data
   ```bash
   python main.py --collect --limit 5000 --window year
2. Build and analyze graphs
   ```bash
   python main.py --analyze
3. visulise
   ```bash
   python main.py --plot
## Screenshots
![image](https://github.com/user-attachments/assets/2325b79f-d954-4d60-8285-a61d56a926d3)
![image](https://github.com/user-attachments/assets/5bbea7fe-f88b-4120-b657-e47cbef43af5)
![image](https://github.com/user-attachments/assets/b8776159-f127-49f6-925e-f791a015f807)

## Configuration
- Subreddits: Modify the list in GraphBuilder.__init__ or pass via command-line.
- Time windows: Adjust the --window parameter or build_temporal_graphs arguments.
- Radiality: Tweak r_min/r_max in draw_* methods to change peripheral spread.

## Acknowledgments
- PRAW for the Reddit API wrapper
- NetworkX for graph algorithms
- python-louvain for community detection
- TextBlob and NLTK for text processing


