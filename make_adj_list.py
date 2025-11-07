import networkx as nx
from collections import defaultdict
import json
import os
from tqdm import tqdm

class Separator:
    def __init__(self, graph: nx.MultiDiGraph):
        # Initialize with a graph and a nested defaultdict for messages
        self.messages = defaultdict(lambda: defaultdict(list))
        self.graph = graph

    def separate(self):
        # Iterate over all edges in the graph
        for u, v, data in tqdm(self.graph.edges(data=True), desc="Processing edges"):
            # For each edge, store its data (excluding 'id') in the messages dict
            self.messages[u][v].append({key: value for key, value in data.items() if key != 'id'})
        return self.messages

    def dump(self, sourceFolder):
        # Ensure the folder path ends with a slash
        if sourceFolder[-1] != "/":
            sourceFolder += "/"
        path = sourceFolder + 'messages.json'

        # If file exists, increment filename to avoid overwriting
        counter = 1
        while os.path.exists(path):
            path = sourceFolder + f"messages({counter}).json"
            counter += 1

        assert(not os.path.exists(path))  # Ensure the file does not exist

        # Write messages dict to JSON file
        with open(path, 'w') as json_file:
            json.dump(self.messages, json_file, indent=4)
        
if __name__ == "__main__":
    import argparse
    # Parse command line argument for input file path
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Input path to .gexf file")
    parser = parser.parse_args()
    # Read graph from .gexf file and process it
    messages = Separator(nx.read_gexf(parser.path))
    messages.separate()
    # Dump messages to JSON file in the same folder as input
    messages.dump(parser.path[:parser.path.rfind("/") + 1])
