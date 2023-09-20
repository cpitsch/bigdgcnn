# Torch
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data

# PM4Py
from pm4py import format_dataframe, convert_to_event_log
import pm4py.util.xes_constants as xes
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

# Misc
import pandas as pd
from pathlib import Path
import networkx as nx
from typing import List, Tuple
import numpy as np
from statistics import mean, stdev
from tqdm.auto import tqdm

# Internal
from bigdgcnn.data_processing.instance_graphs import discover_instance_graphs_big
from bigdgcnn.data_processing import make_training_data, discover_model_imf
from bigdgcnn.util import add_artificial_start_end_events


class BIG_Instancegraph_Dataset_With_Attributes(InMemoryDataset):
    def __init__(
        self,
        eventlog: EventLog,
        logname: str,
        root="./data",
        transform=None,
        process_model: Tuple[PetriNet, Marking, Marking]=None,
        imf_noise_thresh: float=0,
        force_reprocess=False,
        case_level_attributes: List[str]=None,
        event_level_attributes: List[str]=None,
        z_score_normalize: bool=False,
    ):
        """

        Args:
            eventlog (EventLog): The event log to be used.
            logname (str): An identifier of the event log. Used for caching the processed data.
            root (str, optional): The root directory to cache/retrieve the dataset. Defaults to "./data".
            transform (_type_, optional): Defaults to None.
            process_model (Tuple[PetriNet, Marking, Marking], optional): The discovered process model. If `None`, one will be discovered using IMf with the `imf_noise_threshold`. Defaults to None.
            imf_noise_thresh (float, optional): The noise threshold for the IMf algorithm. Only relevant if process_model is None. Defaults to 0.
            force_reprocess (bool, optional): Force-reprocess the dataset. Defaults to False.
            case_level_attributes (List[str], optional): The case level attributes to be used as features. Defaults to None.
            event_level_attributes (List[str], optional): The event level attributes to be used as features. Defaults to None.
            z_score_normalize (bool, optional): Whether to z-score normalize the numerical attributes. Defaults to False.
        """
        self.z_score_normalize = z_score_normalize
        self.case_level_attributes = case_level_attributes if case_level_attributes is not None else []
        self.case_attribute_vals = {
            attr: list(set(case.attributes[attr] for case in eventlog))
            for attr in self.case_level_attributes
        }

        self.categorical_case_attrs = {attr for attr, vals in self.case_attribute_vals.items() if any(isinstance(val, str) for val in vals)}


        self.case_attr_z_normalization = {
            attr: (mean(case.attributes[attr] for case in eventlog), stdev(case.attributes[attr] for case in eventlog))
            for attr in self.case_level_attributes
            if attr not in self.categorical_case_attrs
        } if self.z_score_normalize else {}

        self.event_level_attributes = event_level_attributes if event_level_attributes is not None else []
        self.event_attribute_vals = {
            attr: list(set(event[attr] for case in eventlog for event in case))
            for attr in self.event_level_attributes
        }
        self.categorical_event_attrs = {attr for attr, vals in self.event_attribute_vals.items() if any(isinstance(val, str) for val in vals)}

        self.event_attr_z_normalization = {
            attr: (mean(event[attr] for case in eventlog for event in case), stdev(event[attr] for case in eventlog for event in case))
            for attr in self.event_level_attributes
            if attr not in self.categorical_event_attrs
        } if self.z_score_normalize else {}

        if process_model is None:
            # Otherwise, a model is already discovered, and adding artificial events will make it not fitting
            self.eventlog = add_artificial_start_end_events(eventlog)
        else:
            self.eventlog = eventlog
        # I did the activities_index wrong. Rerun tomorrow with import bigdgcnn.datasets as datasets; from pm4py import read_xes; log = read_xes("./helpdesk.xes.gz"); hds = datasets.BIG_Instancegraph_Dataset(log, "helpdesk")
        self.activities_index = list(sorted(list(set(evt[xes.DEFAULT_NAME_KEY] for case in self.eventlog for evt in case))))
        self.logname = logname
        self.activities_index
        self.process_model = process_model
        self.imf_noise_thresh = imf_noise_thresh
        self.force_reprocess = force_reprocess
        self.savefile = f"data_{self.logname}_with_attrs.pt"

        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # self.num_features = 1 + len(self.activities_index)
        # print(self.num_features)
        # self.num_node_features = self.num_features
        # self.num_edge_features = 0
        # self.num_classes = len(self.activities_index)

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

        data_list = [self._networkx_to_pytorch_graph(graph, label) for graph, label in tqdm(data, desc="Converting networkx graphs to PyG Graphs")] # Convert each graph to a pytorch geometric graph. the label is now stored as graph.y

        # Save the data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _make_feature_vector(self, node, caseid):
        idx, activity = node
        ret = np.zeros(len(self.activities_index)+1)
        ret[0] = idx
        ret[self.activities_index.index(activity)+1] = 1

        if len(self.case_level_attributes) > 0 or len(self.event_level_attributes) > 0:
            # Find the case
            case = next((case for case in self.eventlog if case.attributes[xes.DEFAULT_NAME_KEY] == caseid), None)
            if case is None:
                raise ValueError(f"Case {caseid} not found in event log")


        if len(self.case_level_attributes) > 0:
            # Add attributes from data
            case_level_attrs = [case.attributes[attr] for attr in self.case_level_attributes]

            case_level_attrs = np.array([])
            for attr in self.case_level_attributes:
                val = case.attributes[attr] 
                if attr not in self.categorical_case_attrs:
                    if self.z_score_normalize:
                        m, std = self.case_attr_z_normalization[attr]
                        case_level_attrs = np.append(case_level_attrs, (val- m) / std)
                    else:
                        case_level_attrs = np.append(case_level_attrs, val)
                else:
                    np.concatenate((case_level_attrs, self._one_hot_encode(val, self.case_attribute_vals[attr])))

            ret = np.concatenate((ret, case_level_attrs))

        if len(self.event_level_attributes) > 0:
            event = case[idx]
            event_level_attrs = np.array([])
            for attr in self.event_level_attributes:
                val = event[attr] 
                if attr not in self.categorical_event_attrs:
                    if self.z_score_normalize:
                        m, std = self.event_attr_z_normalization[attr]
                        event_level_attrs = np.append(event_level_attrs, (val - m) / std)
                    else:
                        event_level_attrs = np.append(event_level_attrs, val)
                else:
                    np.concatenate((event_level_attrs, self._one_hot_encode(val, self.event_attribute_vals[attr])))

            ret = np.concatenate((ret, event_level_attrs))

        return ret

    def _one_hot_encode(self, value, index):
        ret = np.zeros(len(index))
        ret[index.index(value)] = 1
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
        ]), dtype=torch.long)
        x = torch.tensor(np.array([self._make_feature_vector(node, graph.caseid) for node in sorted(graph.nodes, key=lambda v: v[0])]), dtype=torch.float)
        data = Data(
            x=x,
            edge_index=edge_index.t().contiguous(),
            y=self._one_hot_encode(label, self.activities_index),
        )
        return data     