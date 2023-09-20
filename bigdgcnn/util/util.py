from typing import Any

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog, Event
from pm4py.utils import xes_constants as xes

from datetime import datetime, timedelta
from tabulate import tabulate

def importLog(logpath:Any, verbose:bool=True)->EventLog:
    """A wrapper for PM4Py's log importing function.

    Args:
        logpath (Any): The path to the event log file. Only XES Files supported.
        verbose (bool, optional): Configures if a progress bar should be shown. Defaults to True.

    Returns:
        EventLog: The imported event log.
    """    

    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: verbose}
    return xes_importer.apply(logpath, variant=variant, parameters=parameters)

def add_artificial_start_end_events(log: EventLog, start_act: str = "[start]", end_act: str = "[end]", activityName_key:str = xes.DEFAULT_NAME_KEY) -> EventLog:
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

def add_artificial_start_event(log: EventLog, start_act: str="[start]", activityName_key:str = xes.DEFAULT_NAME_KEY) -> EventLog:
    for trace in log:
        # Add [start] and [end] events to the trace.
        start_time = trace[0][xes.DEFAULT_TIMESTAMP_KEY]

        if type(start_time) is int:
            new_start = start_time - 1
        elif type(start_time) is datetime:
            new_start = start_time - timedelta(seconds=1)
        else:
            new_start = start_time

        start_event = Event({activityName_key: start_act, xes.DEFAULT_TIMESTAMP_KEY: new_start})
        trace._list = [start_event, *trace._list]

    return log

def print_log_statistics(log: EventLog):
    avg_case_length = sum(len(case) for case in log)/len(log)
    num_cases = len(log)
    num_activities = len(list(set(evt[xes.DEFAULT_NAME_KEY] for case in log for evt in case)))
    print(tabulate([
        ["Average case length", avg_case_length],
        ["Number of cases", num_cases],
        ["Distinct activities", num_activities],
    ], tablefmt="fancy_grid", floatfmt=".2f"))