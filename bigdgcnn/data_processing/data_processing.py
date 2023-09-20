from typing import List, Tuple
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

import pm4py.algo.discovery.inductive.variants.im_f.algorithm as imf
from pm4py.ml import split_train_test
from pm4py.utils import xes_constants as xes
import networkx as nx

from bigdgcnn.data_processing.instance_graphs import discover_instance_graphs_big

def discover_model_imf(log: EventLog, noise_threshold: float=0) -> Tuple[PetriNet, Marking, Marking]:
    """Discover a model using the Inductive Miner Infrequent algorithm.

    Args:
        log (EventLog): The event log to be used.
        noise_threshold (float, optional): The noise threshold to be used. Defaults to 0.
    Returns:
        PetriNet: The discovered model
        Marking: The initial marking
        Marking: The final marking
    """    
    return imf.apply(log, {imf.Parameters.NOISE_THRESHOLD: noise_threshold})

def train_test_split(log: EventLog, train_percentage: float, random: bool = False) -> Tuple[EventLog, EventLog]:
    """Split the Event Log into a Training and a Test Set.

    Args:
        log (EventLog): The Event Log
        train_percentage (float): What percentage of the log should be used for training.
        random (bool, optional): Whether the split should be random (True) or chronological (False). Defaults to True.
    Returns:
        Tuple[EventLog, EventLog]: The Training and Test Set
    """
    if random:
        return split_train_test(log, train_percentage)
    else:
        return (log[:int(train_percentage*len(log))], log[int(train_percentage*len(log)):])
    
def graph_train_test_split_chronological(graph_list: List[Tuple[nx.DiGraph, str]], train_percentage: float):
    return (graph_list[:int(train_percentage*len(graph_list))], graph_list[int(train_percentage*len(graph_list)):])


def make_training_data(log: EventLog, model: Tuple[PetriNet, Marking, Marking], activityName_key:str = xes.DEFAULT_NAME_KEY) -> List[Tuple[nx.DiGraph, str]]:
    """Make the Training Data for the Graph Convolutional Network. Make sure you have added artificial start and end events to the log. \
        These artificial events are not added here, as this change would also have to be made to the model. 

    Args:
        log (EventLog): The Event Log
        model (Tuple[PetriNet, Marking, Marking]): The discovered model
        activityName_key (str, optional): The key for the activity name. Defaults to `concept:name`.
    Returns:
        List[Tuple[nx.DiGraph, str]]: List of tuples containing prefix-graphs and the corresponding next activity
    """

    instance_graphs = discover_instance_graphs_big(log, model, activityName_key)
    training_data = []
    for instance_graph in instance_graphs:
        if len(list(nx.simple_cycles(instance_graph))) > 0:
            continue # Skip graphs containing cycles. TODO: Must be due to an error in the instance graph discovery.
        nodes = instance_graph.nodes
        # Sort nodes by index, Then build the prefix graphs for 
        for g, node in enumerate(sorted(nodes, key=lambda node: node[0])[2:], start=2):
            node_index, node_label = node

            # Build subgraph
            allowed_nodes = set(node for node in nodes if node[0] < g)
            subgraph = instance_graph.subgraph(allowed_nodes)
            subgraph.caseid = instance_graph.caseid
            training_data.append((subgraph, node_label))
    return training_data