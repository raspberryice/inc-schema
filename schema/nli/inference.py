from typing import List, Dict, Tuple 
from collections import defaultdict 

import numpy as np
from tqdm import tqdm
from transformers import BartForSequenceClassification, BartTokenizer

from schema.base import Event, Participant 

class InferenceModel(object):
    def __init__(self, xpo_node_dict:Dict, json_events:Dict, model_name:str):
        self.xpo_node_dict = xpo_node_dict
        self.json_events = json_events

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForSequenceClassification.from_pretrained(model_name)
        self.labels = ["contradiction", "neutral", "entailment"]

    def check_inference(self, xpo_event:Dict, premise:str):
        """
        Returns:
            first element as similarity score, second element as label
        """
        
        hypothesis = f'This event is about {xpo_event["name"]}'
        tokens = self.tokenizer(premise, hypothesis, return_tensors="pt")
        outputs = self.model(**tokens)
        logits = outputs.logits
        entail_contradiction_logits = logits
        probs = entail_contradiction_logits.softmax(dim=1)
        index = np.argmax(probs.detach().numpy())
        cur_label = self.labels[index]
        probs_np = probs.detach().numpy()
        score = logits.detach().numpy()[0][2] # get the entailment score 
        return {"score": float(score)}
    

    def get_result(self, events: Dict[int, Event], xpo_name_response: Dict[int, Tuple],
        CURATION_PRIORITY:float=0.1)-> Dict:
        '''
        :param curation_priority: the boost we give to XPO team curated nodes.
        
        '''
        result_dict = defaultdict(dict)

        for idx, evt in tqdm(events.items()):

            event_name_candidate, _ = xpo_name_response[idx]
            # terminate if no candidate found
            if len(event_name_candidate) == 0:
                continue
            # calculate sim score between candidate and event description
            target_xpo_index = -1 
            best_score = None 

            for candidate in event_name_candidate:
                xpo_index = self.xpo_node_dict[candidate][0]
                need_filter = self.json_events[xpo_index]["curated_by"] != "xpo team"
                #check inference model
                model_result= self.check_inference(xpo_event=self.json_events[xpo_index],
                                                   premise=evt.description)
                if model_result is None:
                    continue
                
                score = model_result["score"] 
                if not need_filter:
                    score += CURATION_PRIORITY
                
                if not best_score or score >= best_score:
                    best_score = score 
                    target_xpo_index = xpo_index

            # save result xpo node information
            evt.wd_node = self.json_events[target_xpo_index]["wd_node"]
            evt.wd_name = self.json_events[target_xpo_index]["name"]
            evt.wd_description = self.json_events[target_xpo_index]["wd_description"]

            result_dict[idx]["result_xpo_node"] = {"name": self.json_events[target_xpo_index]["name"],
                                                "wd_node":self.json_events[target_xpo_index]["wd_node"],
                                                "wd_description": self.json_events[target_xpo_index]["wd_description"]}

            # save result xpo node argument names
            argument_data = self.json_events[target_xpo_index]["arguments"]
            for argument in argument_data:
                p = Participant(role_name = argument['name'], event_id = evt.id) 
                p.id = len(evt.participants)
                evt.participants[p.id] = p 

        return result_dict