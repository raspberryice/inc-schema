import json
from typing import List, Dict, Tuple,Set 
from collections import defaultdict 



def parse_grounding_response(response: str, json_events: Dict, 
        xpo_node_dict:Dict[str, List], overlay_parent_dict: Dict[str, Set])-> Tuple[List[str], List]:
    
    event_name_candidate = []
    result_dict_response = []
    cur_list = response.strip().split("\n")
    # retrieve event name candidate and save response string to result_dict
    for response_string in cur_list:
        response_string = response_string.split(".")[-1].strip()
        if response_string not in xpo_node_dict:
            result_dict_response.append(response_string + " (x)")
            continue
        else:
            result_dict_response.append(response_string)
            response_string_xpo_index = xpo_node_dict[response_string][0]
            event_name_candidate.append(response_string)
            # append similar nodes
            if "similar_nodes" in json_events[response_string_xpo_index]:
                similar_nodes = json_events[response_string_xpo_index]["similar_nodes"]
                for similar_node in similar_nodes:
                    similar_node_name = similar_node["name"]
                    if similar_node_name in xpo_node_dict:
                        event_name_candidate.append(similar_node_name)
            # append overlay children
            if response_string in overlay_parent_dict:
                event_name_candidate.extend(list(overlay_parent_dict[response_string]))

    return (event_name_candidate, result_dict_response)



def read_xpo_dict(xpo_file_path:str)-> Tuple[Dict, Dict[str, List], Dict[str, Set]]:
    '''
    Load xpo from json file 
    '''
    data = json.load(open(xpo_file_path))
    json_events = data["events"]
    xpo_node_dict = defaultdict(list)      # key as xpo_node_name, vallue as list of xpo index
    overlay_parent_dict = defaultdict(set) # key as overlay_parent name, value as set of child xpo node name
    for index_name, events_value in json_events.items():
        xpo_node_name = events_value["name"]
        # save xpo_node_name and xpo_index pair
        xpo_node_dict[xpo_node_name].append(index_name)
        # add overlay_parents and child pair
        if "overlay_parents" in events_value:
            overlay_parents = events_value["overlay_parents"]
            for overlay_parent in overlay_parents:
                overlay_parent_name = overlay_parent["name"]
                overlay_parent_dict[overlay_parent_name].add(xpo_node_name)
    
    return json_events, xpo_node_dict, overlay_parent_dict
