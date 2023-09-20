from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, Any, List, Set, Tuple

from pm4py.util import xes_constants as xes

from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
import networkx as nx

from tqdm.auto import tqdm

def add_start_end_activities(log: EventLog, start_act: str = "[start]", end_act: str = "[end]", activityName_key:str = xes.DEFAULT_NAME_KEY) -> EventLog:
    from datetime import datetime, timedelta
    for trace in log:
        # Add [start] and [end] events to the trace.
        start_time = trace[0][xes.DEFAULT_TIMESTAMP_KEY]
        end_time = trace[-1][xes.DEFAULT_TIMESTAMP_KEY]

        if type(start_time) is int:
            new_start = start_time - 1
            new_end = end_time + 1
        elif type(start_time) is datetime:
            new_start = start_time - timedelta(seconds=1)
            new_end = end_time + timedelta(seconds=1)
        else:
            new_start = start_time
            new_end = end_time

        start_event = Event({activityName_key: start_act, xes.DEFAULT_TIMESTAMP_KEY: new_start})
        end_event = Event({activityName_key: end_act, xes.DEFAULT_TIMESTAMP_KEY: new_end})
        trace._list = [start_event, *trace._list, end_event]

    return log


def discover_instance_graphs_big(log:EventLog, model: Any, activityName_key:str=xes.DEFAULT_NAME_KEY) -> List[nx.DiGraph]:
    """Discover the instance graphs using the Big algorithm. [Reference](https://www.sciencedirect.com/science/article/abs/pii/S0957417416301877)

    Args:
        log (EventLog): The event log to be used.
        model (PetriNet): The model to be used.
        activityName_key (str, optional): The key to be used for the activity name. Defaults to concept:name.
    Returns:
        List[nx.DiGraph]: A list of the traces and their discovered instance graph. Nodes are labelled as (idx, activity) where idx is the index of the corresponding event in the trace.
    """
    
    # Add index attribute to events. Annoyingly, in some cases of some event logs (e.g., case 53 of the helpdesk log),  there exist events that
    # are exactly identical down to the case id. By adding the index attribute, this makes them discernable without a too large overhaul of the code.
    for case in log:
        for idx, evt in enumerate(case):
            evt["big-algorithm::index"] = idx
    # log = [case for case in log if len(list(set(case._list))) == len(case._list)] # Alternative: filter out all cases that contain duplicate events.

    model_footprint = footprints_discovery.apply(*model)    

    result = []
    for trace in tqdm(log, "Discovering Instance Graphs. Completed Traces"):
        instance_graph = extract_instance_graph(trace, model_footprint, activityName_key)
        i, d = _get_inserted_and_deleted_activities(trace, model)
        if len(d.union(i)) != 0:
            # Not fitting on model.
            instance_graph = irregular_graph_repairing(instance_graph, trace, d, i, model_footprint['sequence'])
        relabeled_ig = nx.relabel_nodes(instance_graph, mapping={evt: (idx, evt[activityName_key]) for idx, evt in enumerate(trace)})
        relabeled_ig.caseid = trace.attributes[xes.DEFAULT_TRACEID_KEY]
        result.append(relabeled_ig)	
    return result

def _get_inserted_and_deleted_activities(trace: Trace, model: PetriNet) -> Tuple[Set[Tuple[str, ...]], List[Tuple[str, ...]]]:
    """Get the inserted and deleted activities in computing an alignment on the model.

    Args:
        trace (Trace): The trace
        model (PetriNet): The Process Model (A Petri net)

    Returns:
        Tuple[List[Tuple[int, List[str]]]]: The lists of inserted and deleted activities (in that order). Each containing lists of consecutive deleted/inserted activities and the index in the case where this occurred. 
    """    
    alignment_result = alignments.apply(trace, *model) #:class:`dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    alignment = alignment_result['alignment']          #The alignment is a sequence of labels of the form (a,t), (a,>>), or (>>,t) representing synchronous/log/model-moves.
    alignment = [step for step in alignment if step[0] is not None and step[1] is not None] # Ignore silent transitions


    d = set()
    i = set()
    current_list = []
    current_trace_index = 0 # We start counting at 0, so the first log move will be 0.s
    current_starting_index = -1
    for idx, step in enumerate(alignment):
        if step[0] == '>>': # Model move => Deleted Activity
            if len(current_list) == 0 or current_list[0][0] == '>>':
                if len(current_list) == 0:
                    current_starting_index = current_trace_index
                current_list.append(step)
            else:
                i.add((current_starting_index, tuple(act for act, _ in current_list)))
                current_starting_index = current_trace_index
                current_list = [step]
        elif step[1] == '>>': # Log move => Inserted Activity
            if len(current_list) == 0 or current_list[0][1] == '>>':
                if len(current_list) == 0:
                    current_starting_index = current_trace_index
                current_list.append(step)
            else:
                d.add((current_starting_index,tuple(act for _, act in current_list)))
                current_starting_index = current_trace_index
                current_list = [step]
            current_trace_index += 1
        else: # Synchronous move => If current_list is nonempty we dump this into the result lists
            if len(current_list) > 0:
                if current_list[0][0] == '>>': # Currently saving inserted activities
                    d.add((current_starting_index, tuple(act for _, act in current_list)))
                else:  # Currently saving deleted activities
                    i.add((current_starting_index, tuple(act for act, _ in current_list)))
                current_starting_index = -1
                current_list = []
            current_trace_index += 1
    if len(current_list) > 0:
        if current_list[0][0] == '>>':
            d.add((current_starting_index, tuple(act for _, act in current_list)))
        else:
            i.add((current_starting_index, tuple(act for act, _ in current_list)))
    return i, d


def extract_instance_graph(trace:Trace, model_footprint: Dict[str, Any], activityName_key:str=xes.DEFAULT_NAME_KEY):
    """Discover the instance graphs using the Big algorithm. [Reference](https://www.sciencedirect.com/science/article/abs/pii/S0957417416301877)

        This part is without the irregular graph repairing. For this, use `discover_instance_graphs_big`.

    Args:
        log (EventLog): The event log to be used.
        model (PetriNet): The model to be used.
        activityName_key (str, optional): The key to be used for the activity name. Defaults to concept:name.
    Returns:
        List[nx.DiGraph]: The discovered instance graphs.
    """

    nodes = {event for event in trace}
    edges = {
        (e1, e2)
        for i1, e1 in enumerate(trace)
        for i2, e2 in enumerate(trace[i1+1:], start=i1+1)
        
        if (e1[activityName_key], e2[activityName_key]) in model_footprint['sequence'] # "such that causal relation exists between the corresponding activities in the model"
        if all((e1[activityName_key], e[activityName_key]) not in model_footprint['sequence'] for e in trace[i1+1: i2]) \
        or all((e[activityName_key], e2[activityName_key]) not in model_footprint['sequence'] for e in trace[i1+1: i2])
    }

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph

def irregular_graph_repairing(instance_graph: nx.DiGraph, trace: Trace, d: set, i: set, model_causal_relations: List[Tuple[str, str]], activityName_key:str=xes.DEFAULT_NAME_KEY):
    """Repair the instance graph of a trace that does not fit on the model.

    Args:
        instance_graph (nx.DiGraph): The instance graph of the trace.
        trace (Trace): The trace.
        d (set): The set of deleted activities (log moves).
        i (set): The set of inserted activities (model moves).
        model (PetriNet): The model (A Petri net).
    """

    new_edges = set(instance_graph.edges)
    for idx, act in d:
        new_edges = deletion_repair(trace, new_edges, idx, act, model_causal_relations, activityName_key)
    for idx, act in i:
        new_edges = insertion_repair(trace, new_edges, idx, act, model_causal_relations, activityName_key)
    
    new_graph = nx.DiGraph()
    new_graph.update(nodes=instance_graph.nodes, edges=new_edges)
    return new_graph

def _get_event_index(case: Trace, event: Event):
    for i, e in enumerate(case):
        if e == event:
            return i
    return -1

def _sequence_in_edges(sequence: Tuple[str,...], edges: Set[Tuple[Event, Event]]):
    return all(edge in edges for edge in zip(sequence, sequence[1:]))

def deletion_repair(trace:Trace, edges: Set[Tuple[str, str]], deletetion_index: int, deleted_activities: List[str], model_causal_relations: List[Tuple[str, str]], activityName_key:str=xes.DEFAULT_NAME_KEY) -> Set[Tuple[str,str]]:
    i = deletetion_index
    
    e_i = trace[i]
    w_r1 = {
        (e_k, e_i)
        for k, e_k in enumerate(trace[:i])
        if (e_k, e_i) in edges
        and any((e_h[activityName_key], deleted_activities[0]) in model_causal_relations for e_h in trace[k:i])
        and (deleted_activities[-1], e_i[activityName_key]) in model_causal_relations # i dont get why this is in here because both are independent of the variables in here. Or is it just to make the set empty if necessary
    }
    w_r2 = {
        (e_k, e_j)
        for k, e_k in enumerate(trace[:i])
        for j, e_j in enumerate(trace[i+1:], start=i+1)
        if (e_k, e_j) in edges
        and (e_k[activityName_key], deleted_activities[0]) in model_causal_relations
        and (deleted_activities[-1], e_i[activityName_key]) in model_causal_relations
        and any((e_l, e_j) in edges for e_l in trace[i+1:j])
    }

    new_edges = edges.difference(w_r1.union(w_r2))
    for k in reversed(range(1,i)):
        e_k = trace[k]
        for j in range(i, len(trace)):
            e_j = trace[j]
            if (e_k[activityName_key], deleted_activities[0]) in model_causal_relations \
            and (deleted_activities[-1], e_j[activityName_key]) in model_causal_relations \
            and not _sequence_in_edges(tuple(evt for evt in trace[k:j+1]), new_edges) \
            and (not any((e_k, e_l) in new_edges for e_l in trace[k+1:j]) \
            or not any((e_m, e_j) in new_edges for e_m in trace[k+1:i])):
                new_edges.add((e_k, e_j))
    return new_edges

def insertion_repair(trace: Trace, edges: Set[Tuple[str, str]], insertion_index: int, inserted_activities: List[str], model_causal_relations: List[Tuple[str, str]], activityName_key:str=xes.DEFAULT_NAME_KEY) -> Set[Tuple[str,str]]:
    i = insertion_index
    j = i + len(inserted_activities) - 1

    w_r1 = {
        (e_k, e_l)
        for (e_k, e_l) in edges
        if _get_event_index(trace, e_k) < i
        if i <= _get_event_index(trace, e_l) <= j 
    }

    w_r2 = {
        (e_k, e_l)
        for (e_k, e_l) in edges
        if i <= _get_event_index(trace, e_k) <= j 
        if j < _get_event_index(trace, e_k)
    }

    w_r3 = {
        (e_k, e_l)
        for (e_k, e_l) in edges
        if i <= _get_event_index(trace, e_k) <= j 
        if i <= _get_event_index(trace, e_l) <= j 
    }

    new_edges = edges.difference(w_r1.union(w_r2).union(w_r3))

    w_a1 = set()
    w_a2 = set()
    for k, e_k in enumerate(trace[j+1:], start=j+1):
        # Had to add i==0 because of list index out of range
        # and (i==0 or trace[i-1][activityName_key], e_k[activityName_key]) in model_causal_relations \
        if (e_k[activityName_key] not in inserted_activities) and ((trace[i-1][activityName_key], e_k[activityName_key]) in model_causal_relations or (trace[i-1], e_k) in edges) \
        and not _sequence_in_edges(tuple(evt for evt in trace[j:k+1]), new_edges): # TODO: I need another i>0 check here
        # and (trace[j], e_k) not in new_edges: # TODO: I need another i>0 check here
            new_edges.add((trace[j], e_k))
            w_a1.add((trace[j], e_k))
    if i>0 and i<len(trace)-1 and (trace[i-1][activityName_key], trace[i+1][activityName_key]) not in model_causal_relations:
        new_edges.add((trace[i-1], trace[i]))
        w_a2.add((trace[i-1], trace[i]))
    else:
        for k in reversed(range(1,i)):
            e_k = trace[k]
            if e_k[activityName_key] not in inserted_activities \
            and ((e_k[activityName_key], trace[j+1][activityName_key]) in model_causal_relations or (e_k, trace[j+1]) in edges) \
            and not _sequence_in_edges(tuple(evt for evt in trace[k:i+1]), new_edges):
                new_edges.add((e_k, trace[i]))
                w_a2.add((e_k, trace[i]))
    w_a3 = { (trace[k], trace[k+1]) for k in range(max(0,i-1), j-1) } # had to add max(0,i-1) because of negative index. Also with j, (k could be j-1 max) i had an index out of bounds
    new_edges = new_edges.union(w_a3)
    w_r4 = {
        (e_k, e_l)
        for (e_k, e_h) in w_a2
        for (e_p, e_l) in w_a1
        if any(evt[activityName_key] == e_h[activityName_key] for evt in trace[i:j+1])
        if any(evt[activityName_key] == e_p[activityName_key] for evt in trace[i:j+1])
    }

    new_edges = new_edges.difference(w_r4)
    if i>0 and i<len(trace)-1 and (trace[i-1][activityName_key], trace[i+1][activityName_key]) not in model_causal_relations: # Had to add i bounds checks
        w_r5 = {(trace[i-1], e_k) for e_k in trace[i+1:]}
        new_edges = new_edges.difference(w_r5)

    return new_edges


if __name__ == "__main__":
    import pm4py
    import networkx as nx
    import matplotlib.pyplot as plt
    log = pm4py.read_xes('./ml_test_log.xes')
    log_discovery = pm4py.read_xes('./ml_log-discovery.xes')
    log = add_start_end_activities(log)
    log_discovery = add_start_end_activities(log_discovery)

    for case in log:
        for idx, evt in enumerate(case):
            evt['event-identifier'] = f"e_{idx}"


    model = pm4py.discover_petri_net_inductive(log_discovery)
    log = log[2:]
    instance_graphs = discover_instance_graphs_big(log, model)
    for i, instance_graph in enumerate(instance_graphs):
        print([evt['concept:name'] for evt in log[i]])
        mapping = {node: node['concept:name'] for node in instance_graph.nodes}
        g = nx.relabel_nodes(instance_graph, mapping)
        nx.draw(g, with_labels=True)
        plt.show()

    # print('#'*240)
    # log = pm4py.read_xes('./their_log.xes')
    # log = add_start_end_activities(log)
    # # model = pm4py.read_pnml('./petri.pnml')
    # model = pm4py.read_pnml('./petri_padded.pnml')
    # for case in log:
    #     for idx, evt in enumerate(case):
    #         evt['event-identifier'] = f"e_{idx}"
    # instance_graphs = discover_instance_graphs_big(log, model)
    # for case, ig in zip(log, instance_graphs):
    #     print([evt["concept:name"] for evt in case])
    #     fig, ax = plt.subplots()
    #     # nx.draw(ig, with_labels=True)
    #     # nx.draw(nx.relabel_nodes(ig, {node: node['concept:name'] for node in ig.nodes}), with_labels=True)
    #     relabeled = nx.relabel_nodes(ig, {node: (node['event-identifier'], node['concept:name']) for node in ig.nodes})
    #     pos = nx.spring_layout(relabeled)
    #     nx.draw(relabeled, with_labels=True, pos=pos, ax=ax)
    #     ax.set_title("<" + ','.join([evt["concept:name"] for evt in case]) + ">")
    #     plt.show()