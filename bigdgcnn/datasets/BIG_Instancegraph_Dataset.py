# Torch
import torch
from torch_geometric.data import InMemoryDataset, Data

# PM4Py
import pm4py.util.xes_constants as xes
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

# Misc
import networkx as nx
from typing import Tuple
import numpy as np

# Internal
from bigdgcnn.data_processing import make_training_data, discover_model_imf
from bigdgcnn.util import add_artificial_start_end_events



class BIG_Instancegraph_Dataset(InMemoryDataset):
    def __init__(self, eventlog: EventLog, logname: str, root="./data", transform=None, process_model: Tuple[PetriNet, Marking, Marking]=None, imf_noise_thresh: float=0, force_reprocess=False):
        """

        Args:
            eventlog (EventLog): The event log to be used.
            logname (str): An identifier of the event log. Used for caching the processed data.
            root (str, optional): The root directory to cache/retrieve the dataset. Defaults to "./data".
            transform (_type_, optional): Defaults to None.
            process_model (Tuple[PetriNet, Marking, Marking], optional): The discovered process model. If `None`, one will be discovered using IMf with the `imf_noise_threshold`. Defaults to None.
            imf_noise_thresh (float, optional): The noise threshold for the IMf algorithm. Only relevant if process_model is None. Defaults to 0.
            force_reprocess (bool, optional): Force-reprocess the dataset. Defaults to False.
        """        
        if process_model is None:
            # Otherwise, a model is already discovered, and adding artificial events will make it not fitting
            self.eventlog = add_artificial_start_end_events(eventlog)
        else:
            self.eventlog = eventlog

        self.activities_index = list(sorted(list(set(evt[xes.DEFAULT_NAME_KEY] for case in self.eventlog for evt in case))))
        self.logname = logname
        self.activities_index
        self.process_model = process_model
        self.imf_noise_thresh = imf_noise_thresh
        self.force_reprocess = force_reprocess
        self.savefile = f"data_{self.logname}.pt"

        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [] # No need for raw files as the event log is given in the constructor

    @property
    def processed_file_names(self):
        return [self.savefile] if not self.force_reprocess else ["force-reprocess"]
    

    def process(self):
        # If necessary, discover the model
        if self.process_model is None:
            self.process_model = discover_model_imf(self.eventlog, self.imf_noise_thresh)

        self.activities_index = list(sorted(list(set(evt[xes.DEFAULT_NAME_KEY] for case in self.eventlog for evt in case))))

        data = make_training_data(self.eventlog, self.process_model, xes.DEFAULT_NAME_KEY) # Get all prefix graphs of all instance graphs in the event log, together witth their label

        data_list = [self._networkx_to_pytorch_graph(graph, label) for graph, label in data] # Convert each graph to a pytorch geometric graph. the label is now stored as graph.y

        # Save the data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _make_feature_vector(self, node):
        idx, activity = node
        ret = np.zeros(len(self.activities_index)+1)
        ret[0] = idx
        ret[self.activities_index.index(activity)+1] = 1
        return ret

    def _one_hot_encode(self, activity):
        ret = np.zeros(len(self.activities_index))
        ret[self.activities_index.index(activity)] = 1
        return torch.tensor(ret)

    def _networkx_to_pytorch_graph(self, graph: nx.DiGraph, label: str) -> Data:
        """Convert a networkx graph to a pytorch geometric graph. Similar to `torch_geometric.utils.from_networkx`.

        Args:
            graph (nx.DiGraph): The NetworkX DiGraph
            label (str): The label for the graph. I.e., the next activity.
        """
        edge_index = torch.tensor(np.array([
            [edge[0][0], edge[1][0]] # Use the index of the node in the case as the edge index
            for edge in graph.edges
        ]), dtype=torch.int64)
        x = torch.tensor(np.array([self._make_feature_vector(node) for node in sorted(graph.nodes, key=lambda v: v[0])]), dtype=torch.float32)
        data = Data(
            x=x,
            edge_index=edge_index.t().contiguous(),
            y=self._one_hot_encode(label),
        )
        return data     