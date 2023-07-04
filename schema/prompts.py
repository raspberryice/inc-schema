from typing import Dict, List, Optional 
from base import Event 

def build_grounding_prompt(evt:Event) -> str:
    in_context_examples = '''
    List event names related to the event "People are infected with this disease":
    1.infection
    2.epidemic
    3.pandemic

    List event names related to the event "It was a robbery-related incident":
    1.robbery
    2.burglary
    3.theft

    List event names related to the event "The first case of the disease have detected and it has been reported":
    1.infection
    2.epidemic
    3.pandemic

    List event names related to the event "The disease is eventually brought under control":
    1.control
    2.improvement

    List event names related to the event "People who are ill have serious symptoms":
    1.symptoms


    List event names related to the event "The pathogen begins to spread through the population":
    1.transmission
    2.spread

    '''
    prompt = f'List event names related to the event "{evt.description}": \n'


    return in_context_examples + prompt 
 

def build_expansion_prompts(evt: Event)-> Dict[str, str]:
    subevent_during_prompt = f'What happened during "{evt.description}"? List the answers: '
    subevent_step_prompt = f'What are the steps in "{evt.description}"? List the answers: '
    temporal_after_prompt = f'What happens after "{evt.description}"? List the answers: '
    temporal_before_prompt = f'What happened before "{evt.description}"? List the answers: '
    causal_after_prompt = f'List the consequences of "{evt.description}":'
    causal_before_prompt = f'List the possible causes of "{evt.description}":'
    prompts = {
        'subevent_during': subevent_during_prompt,
        'subevent_step': subevent_step_prompt,
        'causal_after': causal_after_prompt,
        'causal_before': causal_before_prompt,
        'temporal_after': temporal_after_prompt,
        'temporal_before': temporal_before_prompt, 
    }

    return prompts


def build_skeleton_prompt(scenario:str, evt: Optional[Event]=None)->str:
    if evt!=None:
        # return f"List the major events that happen in the'{evt.description}' of a {scenario}:"
        return f'''
        {evt.name} is defined as "{evt.description}". \n
        List the major events that happen in the {evt.name} of a {scenario}:
        '''
    else:
        return f"List the major events that happen in a {scenario}:"


def build_retrieval_prompt(retrieved: Dict, prompt:str) -> str:
    return  "Based on the following passages: " + '\n'.join(retrieved.values()) + '\n' + prompt 

def build_relevance_prompt(evt:Event, passage:str)->str:
    return f"Given the passage: {passage} \n Is the following event '{evt.description}' related to the passage? Answer yes or no."

def build_specificity_prompt(evt: Event) -> str: 
    EXAMPLES = '''
    Does the text contain any specific names, numbers, locations or dates? Answer yes or no. 

    Text: The UN Strategy for Recovery is launched in an attempt to rebuild the areas most affected by the Chernobyl disaster. Answer: Yes
    Text: More than 300 teachers in the Jefferson County school system took advantage of counseling services. Answer: Yes 
    Text: The police or other law enforcement officials will interview witnesses and potential suspects. Answer: No
    Text: The IHT will establish a Defense Office to ensure adequate facilities for counsel in the preparation of defense cases. Answer: Yes
    Text: Helping people to recover emotionally and mentally from the trauma of the disaster. Answer: No
    Text: The area is cleaned up and any contaminated materials are removed. Answer: No
    Text: About 100,000 people evacuated Mariupol. Answer: Yes
    Text: Gabriel Aduda said three jets chartered from local carriers would leave the country on Wednesday. Answer: Yes
    Text: The party attempting the coup becomes increasingly frustrated with the ruling government. Answer: No
    Text: The international community condemns the war and calls for a peaceful resolution: Answer: No
    '''

    return f"{EXAMPLES}\n Text:{evt.description} Answer: "

def build_naming_prompt(evt:Event)->str:
    NAMING_EXAMPLES = '''
    Give names to the described event. 
    Description: Disinfect the area to prevent infection of the disease. Name: Sanitize 
    Description: A viral test checks specimens from your nose or your mouth to find out if you are currently infected with the virus. Name: Test for Virus  
    Description: If the jury finds the defendant guilty, they may be sentenced to jail time, probation, or other penalties. Name: Sentence
    Description: The police or other law enforcement officials arrive at the scene of the bombing. Name: Arrive at Scene
    Description: The attacker parks the vehicle in a location that will cause maximum damage and casualties.  Name: Park Vehicle
    Description: The government declares a state of emergency. Name: Declare Emergency
    Description: The government mobilizes resources to respond to the outbreak. Name: Mobilize Resources
    Description: The liable party is required to pay damages to the affected parties. Name: Pay Damages
    Description: People declare candidacy and involve in the campaign for party nomination. Name: Declare Candidacy 
    Description: Assessing the damage caused by the disaster and working on a plan to rebuild. Name: Assess Damage
    '''

    return f"{NAMING_EXAMPLES}\n Description:{evt.description} Name: "


# TODO: add in-context examples 
def build_chapter_multichoice_prompt(evt:Event, chapter_evts: List[Event]):
    prefix = f'Which chapter does "{evt.description} belong to? Options: "'
    chapter_names = [c.name for c in chapter_evts]
    options = '\t'.join(chapter_names)
    return prefix + options 

def build_chapter_binary_prompt(evt: Event, chapter_evt: Event):
    
    prompt = f'''
    {chapter_evt.name} is defined as "{chapter_evt.description}"
    {evt.name} is defined as "{evt.description}" 
    Is {evt.name} a part of {chapter_evt.name}? Answer yes or no. 
    '''

    return prompt 

def build_verification_decomp_prompt(e1: Optional[Event], e2: Optional[Event])-> Dict[str, str]:
    '''
    For use with `verify_edges_decomposition`.
    '''
    if e1 == None:
        d1 = "N/A"
    else:
        d1 = e1.description
    if e2 == None:
        d2 = "N/A"
    else:
        d2 = e2.description 
    
    start_prompt = f'Does "{d1}" start before "{d2}"? Answer yes, no or unknown.'
    end_prompt = f'Does "{d1}" end before "{d2}"? Answer yes, no or unknown.'
    duration_prompt = f'Is the duration of {d1} longer than {d2}? Answer yes or no.'

    prompts = {
        'start': {
            'prompt': start_prompt,
            'vocab': [
                'yes',
                'unknown',
                'no'
            ]
            },
        'end': {
            'prompt': end_prompt,
            'vocab': [
                'yes',
                'unknown',
                'no'
            ]
            },
        'duration':{
            'prompt': duration_prompt,
            'vocab': [
                'yes',
                'no'
            ]
            }
    }
    return prompts 



def build_verification_prompt(e1:Optional[Event], e2: Optional[Event])-> Dict[str, Dict]:
    '''
    e1 is potentially the higher level.
    '''
    if e1 == None:
        d1 = "N/A"
    else:
        d1 = e1.description
    if e2 == None:
        d2 = "N/A"
    else:
        d2 = e2.description 
    
    temporal_prompt = f'Does "{d1}" happen before or after "{d2}"? Answer before, after or unknown.'
    hierarchy_prompt = f'Is "{d2}" a part of "{d1}"? Answer yes, no or unknown.'
    # hierarchy_prompt = f'Does "{d2}" happen during "{d1}"? Answer yes, no or unknown.'

    prompts = {
        'hierarchy': {
            'prompt': hierarchy_prompt,
            'vocab': [
                'yes',
                'unknown',
                'no'
            ]
            },
        'temporal': {
                'prompt': temporal_prompt,
                'vocab': [
                    'before',
                    'unknown',
                    'after'
                ]
            },
    }
    return prompts 