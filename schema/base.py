import os 
import json 
from typing import List, Dict, Tuple, Optional, Union, Set
import re 
from collections import defaultdict 
from copy import deepcopy 
from functools import partial # partial does NOT WORK with jsonpickle objects 
import string 

import networkx as nx 
import torch 
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim 
from jellyfish import levenshtein_distance, jaro_similarity, jaro_winkler_similarity
import jsonpickle 

from log import get_logger 

logger = get_logger('root')

DEFAULT_WEIGHT=0.1
DESC_STRING_SIM_THRES=0.9
NAME_LEV_SIM_THRES=3 
NAME_STRING_SIM_THRES=0.9
DESC_EMB_SIM_THRES=0.85

class Participant(object):
    def __init__(self, role_name: str, event_id: int) -> None:
        self.id = -1 
        self.event_id = event_id 
        self.role_name = role_name 
        self.entity = None # TODO: this part is not automated yet 
        self.templateParticipant = None 

    def __repr__(self) -> str:
        return self.role_name 

    @property
    def at_id(self) -> str:
        at_id = f'resin:Participants/2{self.event_id:03}{self.id:01}'
        return at_id 

    def to_json(self) -> str:
        obj= {
            "@id": self.at_id,
            "roleName": self.role_name,
            "entity": self.entity.at_id if self.entity else ''
        }

        if self.templateParticipant:
            obj['templateParticipant'] = self.templateParticipant.at_id 
        
        return obj 
        

class Entity(object):
    def __init__(self, name: str) -> None:
        self.id = -1 
        self.name = name 
        self.wd_name = ''
        self.wd_node = ''
        self.wd_description = ''

    def __repr__(self) -> str:
        return self.name 
    
    @property
    def at_id(self):
        at_id = f'resin:Entities/0{self.id:04}'
        return at_id 

    def to_json(self):
        return {
            "name": self.name,
            "@id": self.at_id,
            "wd_node": f'wd:{self.wd_node}',
            "wd_label": self.wd_name,
            "wd_description": self.wd_description 
        }


class Event(object):
    def __init__(self, description: str):
        self.id = -1
        self.description = description
        self.name = '' 
        self.level =0 # 0 for primitive event 

        self.wd_node = ''
        self.wd_description = ''
        self.wd_name = ''

        self.is_schema = False 
        self.is_chapter = False 

        self.children = set() #type: Set[int]
        self.parent = -1 
        self.before = set() #type: Set[int]
        self.after = set() #type: Set[int]

        self.instanceOf = None # type: Union[Event, None]
        self.repeatable = False 
        self.participants = {} # type: Dict[int, Participant] # TODO: change this to a list? 
        self.entities = [] # type: List[ Entity]
        self.relations = [] # type: List[ Relation] 

    def __repr__(self):
        return f'{self.name}:{self.description}'

    @property
    def at_id(self):
        return f'resin:Events/1{self.id:04}'

    def is_primitive(self)->bool:
        return (self.level == 0) 

    def get_descendents(self, event_dict)-> List[int]:
        '''
        recursively collect all descendents.
        '''
        if self.is_primitive():
            return []
        else:
            descendents = list(self.children)
            for c in self.children:
                descendents.extend(event_dict[c].get_descendents(event_dict))
            return descendents 
      
    def get_chapter(self, event_dict:Dict):
        if self.is_chapter: return self 
        elif self.parent !=-1:
            p = event_dict[self.parent]
            return p.get_chapter(event_dict)
        else:
            raise ValueError(f'event {self.name} is not assigned to chapter!')
            

    def update_level(self, events:Dict):
        '''
        recursively update the level of parents.
        '''
        if len(self.children) == 0: 
            self.level =0 
        else: 
            self.level = max([events[eid].level for eid in self.children]) +1 
        
        if self.parent != -1:
            p = events[self.parent] # type: Event 
            p.update_level(events) 
        return 
    

    def to_json(self, event_dict: Dict) -> str:
        result = {
            "@id": self.at_id,
            "name":self.name,
            "description": self.description,
            "wd_node": self.wd_node,
            "wd_label": self.wd_name, 
            "wd_description": self.wd_description,
            "isSchema": True if (self.is_schema or self.is_chapter) else False,
            "repeatable": self.repeatable,
        }

        result['outlinks'] = [event_dict[evt_id].at_id for evt_id in self.after]
        result['participants'] = [p.to_json() for p in self.participants.values()]
        
        # children 
        if len(self.children)>0:
            result['children'] = [event_dict[c].at_id for c in self.children]
            result['children_gate'] = "or" #TODO: logical gates are not automated 
        if self.instanceOf:
            result['instanceOf'] = self.instanceOf.at_id 
        
        result['entities'] = [ent.to_json() for ent in self.entities]
        result['relations'] = [rel.to_json() for rel in self.relations] 
        
        return result 
    
class Relation(object):
    def __init__(self, name: str, subj: Union[Entity, Event], obj: Union[Entity, Event]) -> None:
        self.id = -1 
        self.name = name
        self.relationSubject = subj
        self.relationObject = obj 
        self.wd_node = ''
        self.wd_name = ''
        self.wd_description = '' 
    
    def __repr__(self) -> str:
        return self.name 

    
    @property 
    def at_id(self) -> str:
        at_id = f'resin:Relations/3{self.id:04}'
        return at_id 
    
    def to_json(self) -> str:
        return {
            "@id": self.at_id, 
            "name": self.name,
            "relationSubject": self.relationSubject.at_id,
            "relationObj": self.relationObject.at_id,
            "wd_node": self.wd_node, 
            "wd_label": self.wd_name,
            "wd_description": self.wd_description,
        }

class Schema(Event):
    '''
    A schema is an event that can be instantiated.
    '''
    def __init__(self, scenario: str, embedding_model: SentenceTransformer):
        super().__init__(description = f'{scenario} news story.') 

        self.scenario = scenario
        self.events = {} # type: Dict[int, Event]
        self.event_ids = set() 
        self.event_names = set() 
        self.event_embs = {} # type: Dict[int, torch.Tensor]
        self.embedding_model = embedding_model 
        self.G = nx.DiGraph()


        self.is_schema=True 
        self.is_chapter = True 
        # add root node 
        self.name = string.capwords(self.scenario)
        self.entities = []
        self.add_root()

        self.ACTIONS = {
        'temporal_after': self.add_after_event, 
        'temporal_before': self.add_before_event,
        'causal_after': self.add_after_event,
        'causal_before': self.add_before_event,
        'subevent_during': self.add_child_event,
        'subevent_step': self.add_child_event
        }

        self.VERIFY_ACTIONS = {
            'temporal_after': lambda x,y, w: self.add_before_edge(y,x, w),
            'temporal_before': self.add_before_edge, 
            'subevent': lambda x,y, w: self.add_parent_edge(y,x, w),
            'superevent': self.add_parent_edge 
        }

    def __repr__(self):
        return f'SCHEMA: {self.name}:{self.description}'

    def _check_duplicate(self, new_evt:Event)-> Tuple[bool, torch.FloatTensor]:
        '''
        return True if a highly similar event already exists in the schema.
        '''
        new_embed = self.embedding_model.encode(new_evt.description, convert_to_tensor=True) # type: torch.FloatTensor
        
        for eid, evt in self.events.items():
            if jaro_winkler_similarity(new_evt.description, evt.description) > DESC_STRING_SIM_THRES: 
                print(f"New event {new_evt.description} is duplicate of {evt.id}: {evt.description}")
                return True, new_embed
            elif jaro_winkler_similarity(new_evt.name, evt.name) > NAME_STRING_SIM_THRES or levenshtein_distance(new_evt.name, evt.name)<=NAME_LEV_SIM_THRES: 
                print(f"New event {new_evt.description} is duplicate of {evt.id}: {evt.description}")
                return True, new_embed
            elif cos_sim(new_embed, self.event_embs[eid]).item() > DESC_EMB_SIM_THRES: 
                print(f"New event {new_evt.description} is duplicate of {evt.id}: {evt.description}")
                return True, new_embed 
        
        return False, new_embed

    def contains_event(self, evt:Event)-> bool:
        return (evt.id in self.event_ids)

    def add_root(self):
        event_embed = self.embedding_model.encode(self.description, convert_to_tensor=True) # type: torch.FloatTensor
        self.id = 0 
        self.event_ids.add(0)
        self.event_embs[0] = event_embed
        self.event_names.add(self.name)
        self.events[0] = self 
        self.G.add_node(0, label=self.name)
        logger.info(f'Adding root event {self.name} to schema.')
        return 

    def add_event(self, new_evt:Event, check_dup:bool=True)-> bool:
        '''
        Add event to schema. Return False when the event is detected as a duplicate.
        '''
        dup, event_emb = self._check_duplicate(new_evt) 
        if check_dup and dup == True: return False
        
        new_id = len(self.event_ids)
        new_evt.id = new_id 
        self.event_ids.add(new_id)
        self.event_embs[new_id] = event_emb 
        self.event_names.add(new_evt.name)
        self.events[new_id]=new_evt 
        self.G.add_node(new_id, label=new_evt.name)  # label, shape and color attributes are used for visualization 

        logger.info(f'Adding event {new_evt.name} to schema')
        return True 

    def add_after_event(self, evt: Event, new_evt:Event, **kwargs)->bool:
        assert (evt.id in self.event_ids)
        if new_evt.id == -1:
            outcome  = self.add_event(new_evt, **kwargs)
            if not outcome: return False 

        evt.after.add(new_evt.id)
        new_evt.before.add(evt.id)
        self.G.add_edge(evt.id, new_evt.id, type='temporal', weight=DEFAULT_WEIGHT)

        logger.info(f'Adding temporal edge {evt.name} --> {new_evt.name}')
        return True 

    def add_before_event(self, evt: Event, new_evt: Event, **kwargs)->bool:
        assert (evt.id in self.event_ids)
        if new_evt.id == -1:
            outcome = self.add_event(new_evt, **kwargs)
            if not outcome: return False 

        evt.before.add(new_evt.id)
        new_evt.after.add(evt.id)
        self.G.add_edge(new_evt.id, evt.id, type='temporal', weight=DEFAULT_WEIGHT) 

        logger.info(f'Adding temporal edge {new_evt.name} --> {evt.name}')
        return True 

    def add_child_event(self, evt: Event, new_evt: Event, **kwargs)->bool:
        assert (evt.id in self.event_ids)
        if new_evt.id == -1:
            outcome = self.add_event(new_evt, **kwargs)
            if not outcome: return False 

        evt.children.add(new_evt.id)
        new_evt.parent= evt.id
        evt.update_level(self.events) 

        self.G.add_edge(evt.id, new_evt.id, type='hierarchy', color='cornflowerblue', weight=DEFAULT_WEIGHT)

        logger.info(f'Adding hierarchy edge {evt.name} --> {new_evt.name}')
        return True 

    def add_parent_event(self, evt: Event, new_evt: Event, **kwargs) -> bool:
        assert (evt.id in self.event_ids)
        if new_evt.id == -1:
            outcome = self.add_event(new_evt, **kwargs)
            if not outcome: return False 
        evt.parent = new_evt.id
        new_evt.children.add(evt.id)
        new_evt.update_level(self.events)

        self.G.add_edge(new_evt.id, evt.id, type='hierarchy', color='cornflowerblue', weight=DEFAULT_WEIGHT)
        logger.info(f'Adding hierarchy edge {new_evt.name} --> {evt.name}')
        return True 

    def add_before_edge(self, e1: Event, e2: Event, weight:float):
        '''
        e1 is before e2.
        '''
        e2.before.add(e1.id)
        e1.after.add(e2.id)
        self.G.add_edge(e1.id , e2.id, weight=weight, type='temporal') # will update data if edge exists.
        logger.info(f'Adding temporal edge {e1.name} --> {e2.name}')
        return 

    def delete_before_edge(self, e1:Event, e2:Event):
        e2.before.discard(e1.id)
        e1.after.discard(e2.id)
        self.G.remove_edge(e1.id, e2.id)
        logger.info(f'Removing temporal edge {e1.name} --> {e2.name}')
        return 


    def add_parent_edge(self, e1: Event, e2:Event, weight:float=DEFAULT_WEIGHT, keep_chapter: bool=True):
        '''
        e1 is the parent of e2. 

        :param keep_chapter: if set to True, then chapters will not have new parents 
        '''
        if keep_chapter and e2.is_chapter: return 

        if e2.parent == e1.id:
            # e1 is already the parent of e2, update weight 
            self.G.add_edge(e1.id, e2.id, weight=weight, type='hierarchy')
            return 

        if e2.parent != -1 and self.events[e2.parent].level <= e1.level: 
            return 

        if e2.parent != -1 and self.G.edges[e2.parent, e2.id]['weight'] > weight: 
            return 
        
        # the original parent is more high level and has lower weight, set new parent  
        # remove original parent for e2 and add parent -> e1 -> e2 if e1 does not have parent 
        old_parent = e2.parent 
        if old_parent !=-1:
            self.events[old_parent].children.discard(e2.id)
            self.G.remove_edge(old_parent, e2.id)

            if e1.parent == -1:
                self.events[old_parent].children.add(e1.id) 
                old_weight = self.G.edges[old_parent, e2.id]['weight']
                self.G.add_edge(old_parent, e1.id, type='hierarchy', weight=old_weight)  
                logger.info(f'Adding hierarchy edge {self.events[old_parent].name} --> {e1.name}')
        
        e1.children.add(e2.id)
        e2.parent = e1.id
        self.G.add_edge(e1.id, e2.id, weight=weight, type='hierarchy')
        # set the parent level for e1 and e2's old parent 
        e1.update_level(self.events)

        logger.info(f'Adding hierarchy edge {e1.name} --> {e2.name}')
        
        return 

        
    def visualize(self, outdir:str, suffix: Optional[str]=None, by_chapter:bool=False, max_chapter: int=10, ground: bool=True)->None:
        '''
        :param by_chapter: bool if set to True one image will be produced per chapter. 
        :param max_chapter: int, nodes beyond this will be omitted for clarity in a single graph 
        '''
        scenario_path = self.scenario.replace(' ','_')
        os.makedirs(os.path.join(outdir, scenario_path), exist_ok=True )
        if suffix == None:
            filename = f"{scenario_path}_viz"
        else:
            filename = f"{scenario_path}_{suffix}"
        # set node style 
        # if event is non-primitive use blue
        # if primitive, use yellow 
        for eid in self.G.nodes:
            if self.events[eid].is_schema:
                self.G.nodes[eid]['shape'] = 'rectangle'
                self.G.nodes[eid]['color'] = 'navy'
                self.G.nodes[eid]['fillcolor'] = 'lightblue2'
            elif self.events[eid].is_primitive():
                self.G.nodes[eid]['shape'] = 'rectangle'
                self.G.nodes[eid]['color'] = 'gold'
            else:
                self.G.nodes[eid]['shape'] = 'rectangle'
                self.G.nodes[eid]['color'] = 'navy'
                self.G.nodes[eid]['fillcolor'] = 'lightblue2'
            self.G.nodes[eid]['style'] = "rounded,filled"

            if ground:
                if self.events[eid].wd_name != '':
                    name = self.events[eid].name 
                    wd_name = self.events[eid].wd_name
                    self.G.nodes[eid]['label'] = f'{name}: <{wd_name}>'
            else:
                name = self.events[eid].name 
                self.G.nodes[eid]['label'] = name 

        
        # set edge weight 
        for edge in self.G.edges:
            if self.G.edges[edge]['type'] == 'hierarchy':
                self.G.edges[edge]['color'] = 'cornflowerblue'
            elif self.G.edges[edge]['type'] == 'temporal':
                # only for human assessment graphs, we omit the edge weight 
                weight = self.G.edges[edge].get('weight', DEFAULT_WEIGHT)
                if 'label' in self.G.edges[edge]:
                    del self.G.edges[edge]['label'] 
                self.G.edges[edge]['xlabel'] = f'{weight:.2f}'
                self.G.edges[edge]['color'] = 'black'

        def draw_subgraph(nodes, name):
            chapter_graph = nx.subgraph(self.G, nodes)
            A = nx.nx_agraph.to_agraph(chapter_graph)
            A.layout(prog='dot')
            chapter_path = name.replace(' ','_')
            A.draw(f'{outdir}/{scenario_path}/{filename}_{chapter_path}.png')
            return 

        def output_subgraph_descriptions(chapter_nodes, chapter_name):
            descriptions = {self.events[eid].name: self.events[eid].description for eid in chapter_nodes}
            with open(f'{outdir}/{scenario_path}/{chapter_name}_descriptions.txt','w') as f:
                for n, d in descriptions.items():
                    f.write(f'{n}: {d} \n')

        if by_chapter:
            chapters = {evt for eidx, evt in self.events.items() if evt.is_chapter and not evt.is_schema}
            for c in chapters:
                chapter_nodes = c.get_descendents(self.events)
                chapter_nodes.insert(0, c.id)# put the chapter node at the beginning 

                output_subgraph_descriptions(chapter_nodes, c.name) 
                if len(chapter_nodes) > max_chapter:
                    child_nodes = c.children 
                    displayed = list(child_nodes)
                    displayed.insert(0, c.id) 

                    # only show up to max_chapter nodes 
                    displayed = displayed[:max_chapter]
                    draw_subgraph(displayed, c.name)

                    for eidx in child_nodes:
                        evt = self.events[eidx]
                        nodes= evt.get_descendents(self.events)
                        if len(nodes) > 1:
                            nodes.insert(0, eidx)
                            draw_subgraph(nodes, f'{c.name}_{evt.name}')
                else:
                    draw_subgraph(chapter_nodes, c.name) 

        # single graph 
        A = nx.nx_agraph.to_agraph(self.G)
        # specific shape/color of nodes in 'shape' 'color' and 'label' attributes 
        # A.graph_attr["splines"] = "curved"
        A.layout(prog='dot')
        A.draw(f'{outdir}/{scenario_path}/{filename}.png')
        return 

    def to_schema_json(self)->str: 
        result = {
            "@id": f"resin:Schemas/{self.scenario}",
            "sdfVersion": "3.0",
            "version": "resin:Phase2b",
            "events": [],
        }
        for eid, evt in self.events.items():
            d = evt.to_json(self.events)
            if eid == 0:
                # the schema 
                d['importance'] = []
                d['likelihood'] = [] # TODO: not automated 
            result['events'].append(d)

        return result 


    def save(self, outdir:str):
        scenario_path = self.scenario.replace(' ','_')
        os.makedirs(os.path.join(outdir, scenario_path), exist_ok=True )
        with open(os.path.join(outdir, scenario_path,'graph.pkl'),'wb') as f:
            nx.write_gpickle(self.G, f)

        torch.save(self.event_embs, os.path.join(outdir, scenario_path,'embs.pkl'))

        simple_events = deepcopy(self.events)
        simple_root = simple_events[0]
        simple_root.G = None 
        simple_root.embedding_model = None 
        simple_root.event_embs = None 

        with open(os.path.join(outdir, scenario_path,'events.json'),'w') as f:
            encoded_events = jsonpickle.encode(simple_events, indent=2)
            f.write(encoded_events)

        with open(os.path.join(outdir, scenario_path, f'{scenario_path}.sdf.json'), 'w') as f:
            output = self.to_schema_json()
            json.dump(output, f, indent=2)
        
        return 

    @classmethod 
    def from_file(cls, outdir: str, scenario:str, embedding_model=None):
        scenario_path = scenario.replace(' ','_')
        scenario = scenario.replace('_', ' ')
        if embedding_model: 
            s = Schema(scenario=scenario, embedding_model=embedding_model)
        else:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            s = Schema(scenario=scenario, embedding_model=sbert_model)

        with open(os.path.join(outdir, scenario_path,'events.json'),'r') as f:
            encoded_events = f.read() 
            events = jsonpickle.decode(encoded_events, keys=True, classes=[Event, Participant])

            # convert the str keys to int keys 
            s.events = {int(k): v for k,v in events.items()}

        with open(os.path.join(outdir, scenario_path,'graph.pkl'),'rb') as f:
            s.G = nx.read_gpickle(f)
        
        s.event_embs = torch.load(os.path.join(outdir, scenario_path,'embs.pkl'))
        s.event_ids = set(s.events.keys()) 
        s.event_names = {e.name for e in list(s.events.values())} 

        return s


    