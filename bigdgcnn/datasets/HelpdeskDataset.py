import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import pandas as pd
from pathlib import Path
from pm4py import format_dataframe, convert_to_event_log
import pm4py.util.xes_constants as xes

import networkx as nx
from typing import List

from bigdgcnn.data_processing.instance_graphs import discover_instance_graphs_big
from bigdgcnn.data_processing import make_training_data, discover_model_imf
from bigdgcnn.util import add_artificial_start_end_events

import numpy as np

# FILE_URL = r"https://data.mendeley.com/public-files/datasets/39bp3vv62t/files/20b5d03f-c6f7-4fdc-91c3-67defd4c67bb/file_downloaded"
# FILE_URL = r"https://drive.google.com/file/d/1A0qtQgOaMz_iFtrTcVD61k_UBHKUcWwu/view?usp=sharing"
FILE_URL = r"https://drive.google.com/u/0/uc?id=1A0qtQgOaMz_iFtrTcVD61k_UBHKUcWwu&export=download"

class HelpdeskDataset(InMemoryDataset):
    def __init__(self, root="./cached_datasets/helpdesk", transform=None, process_model=None, force_reprocess=False):
        self.process_model = process_model
        self.force_reprocess = force_reprocess
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['helpdesk.csv'] if not self.force_reprocess else ["force-reprocess"]

    @property
    def processed_file_names(self):
        return ['data.pt'] if not self.force_reprocess else ["force-reprocess"]
    
    def get_caseids(self):
        df = pd.read_csv(self.raw_paths[0])
        return df["CaseID"].unique()
        

    def download(self):
        # Download to `self.raw_dir`.
        download_url(FILE_URL, self.raw_dir, log=True, filename="helpdesk.csv")

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        # Convert DF to pm4py event log
        df = format_dataframe(df, case_id='CaseID', activity_key='ActivityID', timestamp_key='CompleteTimestamp')
        log = convert_to_event_log(df)[:100]

        log = add_artificial_start_end_events(log)
        if self.process_model is None:
            self.process_model = discover_model_imf(log, 0)
        activities_index = list(sorted(list(set(evt[xes.DEFAULT_NAME_KEY] for case in log for evt in case))))
    
        data = make_training_data(log, self.process_model, xes.DEFAULT_NAME_KEY)
        print(len(data))
        data_list = [_networkx_to_pytorch_graph(graph, label, activities_index) for graph, label in data]
        print(len(data_list))
        data, slices = self.collate(data_list)
        print(data)
        torch.save((data, slices), self.processed_paths[0])

def _make_feature_vector(node, activities_index):
    idx, activity = node
    ret = np.zeros(len(activities_index)+1)
    ret[0] = idx
    ret[activities_index.index(activity)+1] = 1
    return torch.tensor(ret)

def _one_hot_encode(activity, activities_index):
    ret = np.zeros(len(activities_index))
    ret[activities_index.index(activity)] = 1
    return torch.tensor(ret)

def _networkx_to_pytorch_graph(graph: nx.DiGraph, label: str, activities_index: List[str]) -> Data:
    """Convert a networkx graph to a pytorch geometric graph. Similar to `torch_geometric.utils.from_networkx`.

    Args:
        graph (nx.DiGraph): The NetworkX DiGraph
        activities_index (List[str]): The list of activities in the log. The same as used for one-hot encoding.
    """    
    edge_index = torch.tensor([
        [edge[0][0], edge[1][0]] # Use the index of the node in the case as the edge index
        for edge in graph.edges
    ], dtype=torch.long)
    x = [torch.tensor(_make_feature_vector(node, activities_index), dtype=torch.float) for node in sorted(graph.nodes, key=lambda v: v[0])]
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=_one_hot_encode(label, activities_index))
    return data
