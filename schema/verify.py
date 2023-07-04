
import json 
import os 
from typing import List, Dict, Tuple, Set, Optional, Callable
from itertools import combinations, permutations 
from collections import defaultdict 
import math 

import numpy as np 
from tqdm import tqdm 

from base import Event, Schema
from prompts import build_verification_prompt, build_verification_decomp_prompt
from gpttools import get_gpt3_score 


from log import get_logger 
logger = get_logger('root', filename='schema.log')

def get_verification_pairs(schema: Schema, chapter_events: List[Event], criteria: str='chapter') -> Set[Tuple[int, int]]:
    verification_pairs = set() # type: Set[Tuple[int, int]]
    if criteria == 'neighbor': # for all neighboring events and 2-hop neighbors 
        for eid, e in schema.events.items():
            if e != schema:
                for nid in schema.G.neighbors(eid): # for a directed graph, this is the successors 
                    if nid == 0: continue # root node 
                    if (eid, nid) not in verification_pairs and (nid, eid) not in verification_pairs:
                        verification_pairs.add((eid, nid))
                    for mid in schema.G.neighbors(nid):
                        if mid == 0: continue # root node 
                        if (mid, eid) not in verification_pairs and (eid, mid) not in verification_pairs:
                            verification_pairs.add((eid, mid)) 
    elif criteria == 'chapter': # for all pairs that belong to the same chapter 
        for c in chapter_events:
            members = c.get_descendents(schema.events) # type: List[int]
            member_pairs = set(combinations(members, r=2)) 
            verification_pairs.update(member_pairs)


    return verification_pairs


def get_edge_score(logprobs: List[Dict], vocab: List[str])-> Dict[str, float]:
    '''
    Take the output from GPT3 and convert the logprobs to a probability distribution over the vocabulary.
    '''
    global_answer_scores = {w: 0.0 for w in vocab} 
    for pred in logprobs: # for each position, aggregate over positions 
        answer_scores =  {w: 0.0 for w in vocab} 
        for tok, score in pred.items(): # for each top token 
            norm_tok = tok.strip().lower() 
            if norm_tok in answer_scores.keys():
                answer_scores[norm_tok] = max(answer_scores[norm_tok], np.exp(score)) 
        for key in global_answer_scores:
            global_answer_scores[key] += answer_scores[key]
    
    # normalize sum to 1 
    prob_sum = sum(v for v in global_answer_scores.values()) 
    norm_answer_scores = {k: v/prob_sum for k,v in global_answer_scores.items()}
    return norm_answer_scores 

def calibrate_score(score: np.ndarray, e1: Event, e2: Event, 
    score_dict:Dict, name: str):
    c1 = np.linalg.inv(np.diag(score_dict[(e1, None)][name]))
    # c2 = np.linalg.inv(np.diag(score_dict[(None, e2)][name]))
    score = c1@ score 
    score /= np.sum(score) # renormalize 
    return score

def get_score_vec(e1:Optional[Event], e2:Optional[Event], prompt_func:Callable, score_dict:Dict):
    p = prompt_func(e1, e2)
    for edge_type, d in p.items():
        score_vec = np.ones((len(d['vocab']),) ) * 1e-4
        try:
            _ , logprobs = get_gpt3_score(d['prompt'])
            edge_score = get_edge_score(logprobs, d['vocab'])
            for i, w in enumerate(d['vocab']):
                score_vec[i] = edge_score[w] 
        except:
            logger.info("gpt3 api error")
    
        score_dict[(e1, e2)][edge_type] = score_vec
    return None 

def verify_edges_decomposition(schema:Schema, verification_pairs: Set[Tuple[int, int]], 
        schema_dir: str, calibrate: bool=False, 
        edge_threshold: float=0.4, duration_threshold: float=0.6) -> Schema:
    
    # compute single event scores for calibration 
    event_scores = defaultdict(dict) # Tuple[int, int] -> str -> np.ndarray (vocab_n, )

    output = {} # for checking results
    for pair in tqdm(verification_pairs):
        e1 = schema.events[pair[0]]
        e2 = schema.events[pair[1]] 
        for tup in [(e1, None), (e2, None), (e1, e2), (e2, e1)]: 
            if tup not in event_scores:
                get_score_vec(tup[0], tup[1], prompt_func=build_verification_decomp_prompt, score_dict=event_scores)

    for pair in verification_pairs:
        e1 = schema.events[pair[0]]
        e2 = schema.events[pair[1]] 
        

        q_start = event_scores[(e1, e2)]['start']
        if calibrate: q_start = calibrate_score(q_start, e1, e2, event_scores, 'start')
        rev_q_start = event_scores[(e2, e1)]['start']
        if calibrate: rev_q_start = calibrate_score(rev_q_start, e2, e1, event_scores, 'start')
        p_start = (q_start + np.flip(rev_q_start, axis=0))/2 


        q_end = event_scores[(e1, e2)]['end']
        if calibrate: q_end = calibrate_score(q_end, e1, e2, event_scores, 'end')
        rev_q_end = event_scores[(e2, e1)]['end']
        if calibrate: rev_q_end = calibrate_score(rev_q_end, e2, e1, event_scores, 'end')
        p_end = (q_end + np.flip(rev_q_end, axis=0))/2

        res = {
            'e1': e1.description,
            'e2': e2.description,
            'q_start': q_start.tolist(),
            'rev_q_start': rev_q_start.tolist(),
            'p_start': p_start.tolist(),
            'q_end': q_end.tolist(),
            'rev_q_end': rev_q_end.tolist(),
            'p_end': p_end.tolist() 
        }

        exist_edge = (pair in schema.G.edges and schema.G.edges[pair]['type']=='temporal')
        if exist_edge or (p_start[0] > edge_threshold and p_end[0] > edge_threshold):
            # e1 ---> e2 
            edge_weight = math.sqrt(p_start[0] * p_end[0])
            schema.VERIFY_ACTIONS['temporal_before'](e1, e2, edge_weight)
            res['rel'] = 'before'
        # elif exist_edge: 
        #     # this temporal edge needs to be removed 
        #     schema.delete_before_edge(e1, e2)

        exist_rev_edge = (reversed(pair) in schema.G.edges and schema.G.edges[reversed(pair)]['type']=='temporal')
        
        # e2 --> e1 
        if exist_rev_edge or (p_start[-1] > edge_threshold and p_end[-1] > edge_threshold):
            edge_weight = math.sqrt(p_start[-1] * p_end[-1])
            schema.VERIFY_ACTIONS['temporal_after'](e1, e2, edge_weight)
            res['rel'] = 'after'
        # elif exist_rev_edge:
        #     schema.delete_before_edge(e2, e1)


        # need to check duration 
        q_duration = event_scores[(e1, e2)]['duration']
        if calibrate: q_duration = calibrate_score(q_duration, e1, e2, event_scores, 'duration')
        rev_q_duration = event_scores[(e2, e1)]['duration']
        if calibrate: rev_q_duration = calibrate_score(rev_q_duration, e2, e1, event_scores, 'duration')
        p_duration = (q_duration + np.flip(rev_q_duration, axis=0)) /2 

        res['q_duration'] = q_duration.tolist() 
        res['rev_q_duration'] = rev_q_duration.tolist() 
        res['p_duration'] = p_duration.tolist() 

        exist_parent_edge = (pair in schema.G.edges and schema.G.edges[pair]['type']=='hierarchy')

        if exist_parent_edge or (p_start[0] > edge_threshold and p_end[-1] > edge_threshold and p_duration[0] > duration_threshold):
            # e1 parent of e2 
            edge_weight = math.sqrt(p_start[0] * p_end[-1])
            schema.VERIFY_ACTIONS['superevent'](e1, e2, edge_weight)
            res['rel'] = 'superevent'

        
        exist_child_edge = (reversed(pair) in schema.G.edges and schema.G.edges[reversed(pair)]['type']=='hierarchy')
        if exist_child_edge or (p_start[-1] > edge_threshold and p_end[0] > edge_threshold and p_duration[-1] > duration_threshold):
            # e2 is the parent of e1 
            edge_weight = math.sqrt(p_start[-1] * p_end[0])
            schema.VERIFY_ACTIONS['subevent'](e2, e1, edge_weight) 

            res['rel'] = 'subevent'


        if 'rel' not in res: res['rel'] = 'none'
        output[str(pair)] = res 


    scenario_path = schema.scenario.replace(' ','_')
    with open(os.path.join(schema_dir,scenario_path, 'ver_decomp_output.json'), 'w') as f:
        json.dump(output, f, indent=2) 
    
    return schema 

    



def verify_edges(schema: Schema, 
        verification_pairs: Set[Tuple[int, int]], 
        schema_dir: str, calibrate: bool=True, 
        edge_threshold: float=0.7, parent_threshold: float=0.7) -> Schema:
    '''
    Verify temporal and hierarchical edges. Prioritizes hierarchical edges over temporal edges.
    '''
    # compute single event scores for calibration 
    event_scores = defaultdict(dict) # Tuple[int, int] -> str -> np.ndarray (vocab_n, )
        
    for pair in verification_pairs:
        e1 = schema.events[pair[0]]
        e2 = schema.events[pair[1]] 
        for tup in [(e1, None), (e2, None), (e1, e2), (e2, e1)]: 
            if tup not in event_scores:
                get_score_vec(tup[0], tup[1], build_verification_prompt, event_scores)

    output = {} 
    # add extra hierarchy edges 
    for pair in verification_pairs:
        e1 = schema.events[pair[0]]
        e2 = schema.events[pair[1]] 
        if e1.level < e2.level: # swap 
            e1, e2 = e2, e1 
    
        score_vec = event_scores[(e1, e2)]['hierarchy']
        if calibrate: score_vec = calibrate_score(score_vec, e1, e2, event_scores, 'hierarchy')

        rev_score_vec = event_scores[(e2, e1)]['hierarchy']
        if calibrate: rev_score_vec = calibrate_score(rev_score_vec, e2, e1, event_scores, 'hierarchy')
        
        
        target_scores = (score_vec + np.flip(rev_score_vec,axis=0)) /2 
        logger.info(f'{e1.name} --> {e2.name} has parent score {target_scores[0]:.2f}')
        
        res = {
            'e1': e1.description,
            'e2': e2.description,
            'q_hier': score_vec.tolist(),
            'rev_q_hier': rev_score_vec.tolist(),
            'p_hier': target_scores.tolist(),
            'rel': [] # could have multiple relations 
        }

        exist_parent_edge = (pair in schema.G.edges and schema.G.edges[pair]['type']=='hierarchy')
        if exist_parent_edge or (target_scores[0] > parent_threshold): 
            schema.VERIFY_ACTIONS['superevent'](e1, e2, target_scores[0])

            res['rel'].append('subevent')
        
        output[str(pair)] = res 


    # add extra temporal edges 
    for pair in verification_pairs:
        e1 = schema.events[pair[0]]
        e2 = schema.events[pair[1]] 
        if e1.parent==e2.id or e2.parent==e1.id: continue 
    
        score_vec = event_scores[(e1, e2)]['temporal']
        if calibrate: score_vec = calibrate_score(score_vec, e1, e2, event_scores, 'temporal')

        rev_score_vec = event_scores[(e2, e1)]['temporal']
        if calibrate: rev_score_vec = calibrate_score(rev_score_vec, e2, e1, event_scores, 'temporal')

        target_scores = (score_vec + np.flip(rev_score_vec, axis=0))/2

        res = {
            'q_temp': score_vec.tolist(),
            'rev_q_temp': rev_score_vec.tolist(),
            'p_temp': target_scores.tolist(),
            'rel': []
        }

        logger.info(f'{e1.name} --> {e2.name} has before score {target_scores[0]:.2f} and after score {target_scores[-1]:.2f}')
        if (pair in schema.G.edges and schema.G.edges[pair]['type']=='temporal') \
            or (target_scores[0] > edge_threshold): 
            edge_weight = target_scores[0]
            schema.VERIFY_ACTIONS['temporal_before'](e1, e2, edge_weight)
            res['rel'].append('before')

        if (reversed(pair) in schema.G.edges and schema.G.edges[reversed(pair)]['type']=='temporal') or \
                (target_scores[-1] > edge_threshold):
            edge_weight = target_scores[-1]
            schema.VERIFY_ACTIONS['temporal_after'](e1, e2, edge_weight)

            res['rel'].append( 'after')
        
        old_res = output[str(pair)] 
        for k, v in res.items():
            if k!= 'rel':
                old_res[k] = v 
            else:
                old_res[k].extend(v)

    
    # logger.info('Saving schema....')
    # schema.visualize(schema_dir, suffix='verification_temp')
    # schema.save(schema_dir)

    scenario_path = schema.scenario.replace(' ','_')
    with open(os.path.join(schema_dir,scenario_path, 'ver_output.json'), 'w') as f:
        json.dump(output, f, indent=2) 
    


    return schema 
