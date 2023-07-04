import os 
import json 
import argparse 
from typing import List, Dict, Set, Tuple 

from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim 
import torch 
import numpy as np 

from base import Schema 
from log import get_logger 
logger = get_logger('root')


def read_reference_schema(filename:str, embedding_model: SentenceTransformer):
    with open(filename,'r') as f:
        ref_schema = json.load(f)
    
    event_list = ref_schema['events']
    event_embs = {} 
    temporal_edge_set = set() 
    hier_edge_set = set() 
    id_set = set() 

    # precompute the embedding for the all the events 
    for evt in event_list:
        eid = evt['@id']
        if 'description' not in evt or evt['description'] == '': 
            raise ValueError(f'{filename}: {eid} does not have an description')
        new_embed = embedding_model.encode(evt['description'], convert_to_tensor=True) # type: torch.FloatTensor
        event_embs[eid] = new_embed 
        id_set.add(eid)


    for evt in event_list:
        eid = evt['@id']
        if 'outlinks' in evt:
            for other_eid in evt['outlinks']:
                if other_eid in id_set:
                    temporal_edge_set.add((eid, other_eid))
        if 'children' in evt:
            for other_eid in evt['children']:
                if other_eid in id_set: 
                    hier_edge_set.add((eid, other_eid))
        
    return event_list, event_embs, temporal_edge_set, hier_edge_set

def create_emb_matrix(event_list:List, event_embs:Dict):
    '''
    event_list: List of ids 
    '''
    tensor_list = []
    for evt in event_list:
        tensor_list.append(event_embs[evt])

    emb_matrix = torch.stack(tensor_list, dim=0)
    return emb_matrix 


def compute_edge_metrics(ref_edges:Set[Tuple], gen_edges: Set[Tuple], gen2ref_assignment:Dict, ref2gen_assignment:Dict):
    # compute edge f1 
    matched_ref = set(gen2ref_assignment.values())
    matched_gen = set(gen2ref_assignment.keys()) 
    correct_n =0 

    matched_ref_edges = set((x,y) for (x,y) in ref_edges if (x in matched_ref and y in matched_ref))
    matched_gen_edges = set((x,y) for (x,y) in gen_edges if (x in matched_gen and y in matched_gen))

    for ref_pair in matched_ref_edges:
        mapped_pair = (ref2gen_assignment[ref_pair[0]], ref2gen_assignment[ref_pair[1]])
        if mapped_pair in matched_gen_edges:
            correct_n += 1 

    if len(matched_ref_edges) ==0: 
        recall =0
    else:
        recall = correct_n/len(matched_ref_edges) 
    if len(matched_gen_edges) ==0:
        prec =0
    else:
        prec = correct_n/ len(matched_gen_edges)
    
    if correct_n == 0: f1 = 0.0
    else:
        f1 = 2/ (1/prec + 1/recall) 

    return prec, recall, f1, len(matched_ref_edges), len(matched_gen_edges)

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation of the schema against human ground truth. All the schemas in the reference dir will be used.') 
    parser.add_argument('--reference_schema_dir',type=str, default='schemalib/phase2b/curated')
    parser.add_argument('--schema_dir', type=str, default='outputs/schema_phase2b_Jan14')
    # parser.add_argument('--scenario', type=str, default='chemical spill')
    args = parser.parse_args() 

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    results = {} 
    assignments= {} 

    for filename in os.listdir(args.reference_schema_dir):
        scenario, ext = os.path.splitext(filename)
        if ext!= '.json': continue 
        scenario = scenario.lower() 
        if scenario == 'chemical_spill': continue # this one is for in-context learning 

        # if scenario == 'coup': continue # this one has some problems with XOR gates 
        

        ref_event_list, ref_embs, ref_temp, ref_hier = read_reference_schema(
            os.path.join(args.reference_schema_dir, filename), sbert_model)

        ref_idx2id = {idx: evt['@id'] for idx, evt in enumerate(ref_event_list)}
        ref_id2idx = {evt['@id']: idx for idx, evt in enumerate(ref_event_list)} 

        ref_N = len(ref_event_list)
        logger.info(f'The reference schema contains {ref_N} events')

        s= Schema.from_file(args.schema_dir, scenario, embedding_model=sbert_model)

        N = len(s.events) 
        logger.info(f'The generated schema contains {N} events')

        results[scenario] = {}

        # build the assignment matrix 
        
        emb_matrix = create_emb_matrix(list(sorted(s.events.keys())), s.event_embs) 
        ref_event_ids= [evt['@id'] for evt in ref_event_list]
        ref_emb_matrix = create_emb_matrix(ref_event_ids, ref_embs)

        sim_matrix = cos_sim(emb_matrix, ref_emb_matrix)
        w = sim_matrix.cpu().numpy() # convert similarity to cost 
        # w = np.zeros((N, ref_N), dtype=np.int64) # cost matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix=w, maximize=True) 

        gen2ref_assignment = {x:ref_idx2id[y] for x,y in zip(row_ind, col_ind)} 
        ref2gen_assignment = {v:k for k,v in gen2ref_assignment.items()}
        # event_id to @id mapping 

        assignments[scenario] = []
        for k,v in gen2ref_assignment.items():
            ref_evt = ref_event_list[ref_id2idx[v]]
            assignments[scenario].append({
                'ref_id': ref_evt['@id'],
                'ref_event': ref_evt['name'],
                'ref_event_description': ref_evt['description'],
                'gen_id': s.events[k].id,
                'gen_event': s.events[k].name,
                'gen_event_description': s.events[k].description
            })


        # the len of row_ind == col_ind == min(N, ref_N)
        score = w[row_ind, col_ind].sum()
        event_prec = score / N 
        event_recall = score / ref_N 

        event_f1 = 2/ (1/event_prec + 1/event_recall)

        print(f'Prec: {event_prec:.3f} Recall: {event_recall:.3f} F1: {event_f1:.3f}')
        results[scenario]['event'] = {
            'prec': event_prec,
            'recall': event_recall,
            'f1': event_f1,
            'ref_n': ref_N,
            'gen_n': N 
        }


        # collect edges from generated schema 
        temp_edges = set()
        hier_edges = set() 
        for eid, evt in s.events.items():
            for other_eid in evt.children:
                hier_edges.add((eid, other_eid))
            
            for other_eid in evt.after:
                temp_edges.add((eid, other_eid))

        temp_prec, temp_recall, temp_f1, ref_n, gen_n = compute_edge_metrics(ref_temp, temp_edges, gen2ref_assignment, ref2gen_assignment)

        print(f'Temporal edge Prec: {temp_prec:.3f} Recall: {temp_recall:.3f} F1: {temp_f1:.3f}')
        results[scenario]['temporal'] = {
            'prec': temp_prec,
            'recall': temp_recall,
            'f1': temp_f1,
            'ref_n': ref_n,
            'gen_n': gen_n
        }

        hier_prec, hier_recall, hier_f1, ref_n, gen_n = compute_edge_metrics(ref_hier, hier_edges, gen2ref_assignment, ref2gen_assignment) 

        results[scenario]['hierarchical'] = {
            'prec': hier_prec,
            'recall': hier_recall,
            'f1': hier_f1,
            'ref_n': ref_n,
            'gen_n': gen_n 
        }

        # print(f'Hierarchical edge Prec: {hier_prec:.3f} Recall: {hier_recall:.3f} F1: {hier_f1:.3f}')



    # average over all scenarios 
    summarized = {
        'event':{
            'prec': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ref_n': 0.0,
            'gen_n': 0.0
        },
        'temporal': {
            'prec': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ref_n': 0.0,
            'gen_n': 0.0
        },
        'hierarchical':
         {
            'prec': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ref_n': 0.0,
            'gen_n': 0.0
        },
    }
    
    for aspect in ['event','temporal','hierarchical']:
        for metric in ['prec','recall','f1','ref_n','gen_n']:
            val = []
            for scenario in results:
                val.append(results[scenario][aspect][metric])
            avg_val = np.mean(np.array(val)) 
            summarized[aspect][metric] = avg_val 
        

    results['total'] = summarized

    with open(os.path.join(args.schema_dir,'eval.json'),'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.schema_dir,'assignments.json'),'w') as f:
        json.dump(assignments, f, indent=2) 

