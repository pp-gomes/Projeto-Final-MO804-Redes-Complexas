import networkx as nx
import os
import argparse
from tqdm import tqdm

class GraphMerger:
    def __init__(self, sourceFolder):
        # Ensure the source folder path ends with a slash
        if not sourceFolder.endswith("/"):
            sourceFolder += "/"
        self.sourceFolder = sourceFolder
        # Initialize an empty MultiDiGraph to store the merged graph
        self.merged_graph = nx.MultiDiGraph()

    def merge(self):
        # Iterate over all files in the source folder
        for filename in os.listdir(self.sourceFolder):
            file_path = os.path.join(self.sourceFolder, filename)
            # Process only .gexf files
            if os.path.isfile(file_path) and file_path.endswith(".gexf"):
                # Read edges from the .gexf file and add them to the merged graph
                for u, v, data in tqdm(nx.read_gexf(file_path).edges(data=True), desc=filename):
                    # Add edge, excluding the 'id' attribute from edge data
                    self.merged_graph.add_edge(u, v,
                        **{key: value for key, value in data.items() if key != 'id'}
                    )

    def save(self):
        # Set the output file path for the merged graph
        folder_path = self.sourceFolder + "reddit_graph_merged.gexf"
        counter = 1
        # If the file already exists, append a counter to the filename
        while os.path.exists(folder_path):
            folder_path = self.sourceFolder + f"reddit_graph_merged({counter}).gexf"
            counter += 1
        # Save the merged graph to a .gexf file
        nx.write_gexf(self.merged_graph, folder_path)

if __name__ == "__main__":
    # Parse command-line arguments for the source folder
    parser = argparse.ArgumentParser(description="Merge multiple graphs into a single .gexf")
    parser.add_argument("--src", type=str, default="./", help="Path of the folder where the graphs are located.\nUse ./ to indicate relative paths instead of global ones")
    args = parser.parse_args()

    # Create a GraphMerger instance and perform merging and saving
    MergedGraph = GraphMerger(args.src)
    MergedGraph.merge()
    MergedGraph.save()
