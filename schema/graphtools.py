'''
Graph algorithms.
'''
from typing import List, Dict, Tuple, Set 

import networkx as nx 

from base import Schema, Event 


def find_feedback_edges(graph, nodes: Set[int], edge_type: str) -> List[Tuple[int, int]]:
    '''
    remove edges that cause loops for the given edge type according to edge weight.
    Implementation of 
    *Eades, P., Lin, X. and Smyth, W.F. (1993) A fast and effective heuristic for the feedback arc set problem. 
    Information Processing Letters, 47 (6). pp. 319-323.*
    '''
    tempgraph = nx.subgraph(graph, nodes).copy() 

   
    extra_edges = [edge for edge in tempgraph.edges if tempgraph.edges[edge]['type']!=edge_type]
    tempgraph.remove_edges_from(extra_edges)
    # sort nodes
    node_order_head = []
    node_order_tail = []

    while True:     
        sink_nodes = []
        source_nodes = []
        for n in tempgraph.nodes:
            if len(list(tempgraph.predecessors(n))) == 0:
                # n is a source node 
                source_nodes.append(n)
            elif len(list(tempgraph.successors(n))) == 0:
                # n is a sink node 
                sink_nodes.append(n)
        if sink_nodes == [] and source_nodes == []: break 
        node_order_head =  node_order_head + source_nodes
        node_order_tail = sink_nodes + node_order_tail 
        tempgraph.remove_nodes_from(source_nodes)
        tempgraph.remove_nodes_from(sink_nodes)
    
    while tempgraph.number_of_nodes() > 0:
        max_delta = -100
        max_node = -1 
        for n in tempgraph.nodes:
            in_score = sum([w for p, n, w in tempgraph.in_edges(n, data='weight')]) 
            out_score = sum([w for n, s, w in tempgraph.out_edges(n, data='weight')])
            delta = out_score - in_score 
            if delta > max_delta:
                max_delta = delta 
                max_node = n 
        node_order_head.append(max_node)
        tempgraph.remove_node(max_node)
    
    node_order = node_order_head + node_order_tail 

    subgraph= nx.subgraph(graph, nodes)

    assert len(node_order) == subgraph.number_of_nodes() 


    node2order = {n: i for i, n in enumerate(node_order)}
    # remove violating edges 
    feedback_edges = []
    for u,v, t in subgraph.edges(data='type'):
        if t == edge_type:
            if node2order[u] > node2order[v]:
                feedback_edges.append((u,v))
    
    return feedback_edges 


def find_transitive_edges(graph, nodes: Set[int], edge_type: str) -> List[Tuple[int, int]]:
    tempgraph = nx.subgraph(graph, nodes).copy() 

   
    extra_edges = [edge for edge in tempgraph.edges if tempgraph.edges[edge]['type']!=edge_type]
    tempgraph.remove_edges_from(extra_edges)
    TR = nx.transitive_reduction(tempgraph) # does not copy attributes 
    transitive_edges = []

    subgraph = nx.subgraph(graph,nodes)
    for u, v, t in subgraph.edges(data='type'):
        if t == edge_type:
            if (u,v) not in TR.edges:
                transitive_edges.append((u,v))
    
    return transitive_edges 

def remove_bad_edges(schema:Schema, chapter_events: List[Event]):
     # remove violating edges
    all_temp_edges = [edge for edge in schema.G.edges if schema.G.edges[edge]['type'] == 'temporal']

    for u,v in all_temp_edges:
        if (u, v) in schema.G.edges: # might be deleted when checking previous edges 

            # cannot exist both temporal and hierarchical relation between events 
            if (v,u) in schema.G.edges and schema.G.edges[v,u]['type'] == 'hierarchy':
                schema.G.remove_edge(u, v)
                schema.events[u].after.discard(v)
                schema.events[v].before.discard(u)

            # cannot exist A->B and B->A edges 
            if (v,u) in schema.G.edges and schema.G.edges[v,u]['type'] == 'temporal':
                w = schema.G.edges[u,v]['weight']
                rev_w = schema.G.edges[v,u]['weight']
                if w> rev_w:
                    schema.G.remove_edge(v, u)
                    schema.events[v].after.discard(u)
                    schema.events[u].before.discard(v)
                else:
                    schema.G.remove_edge(u, v)
                    schema.events[u].after.discard(v)
                    schema.events[v].before.discard(u)
        
        
    # remove loops 
    for c in chapter_events:
        members = c.get_descendents(schema.events) # type: List[int]
        if len(members) <= 2: continue 
        feedback_edges = find_feedback_edges(schema.G, members, edge_type='temporal') 
        for u, v in feedback_edges:
            schema.G.remove_edge(u, v)
            schema.events[u].after.discard(v)
            schema.events[v].before.discard(u)
        # transitive reduction
        transitive_edges = find_transitive_edges(schema.G, members, edge_type='temporal')
        for u, v in transitive_edges:
            schema.G.remove_edge(u, v)
            schema.events[u].after.discard(v)
            schema.events[v].before.discard(u)


   
    return schema 

