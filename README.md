# inc-schema
Code for paper "Open-Domain Hierarchical Event Schema Induction by Incremental Prompting and Verification"


The schema data structure is defined in `base.py`. 
Prompts are available in `prompts.py`.
The entry point for schema induction is `main.py`.



Before you run the model, you will need to put your OpenAI API key in `configs/local.yaml`.

Options can either be specified through a YAML config file (see `configs` directory) or through the command line. 


You can run the model by the following command 
```bash
python schema/main.py --load_config=<your yaml file> --run_name=<your run name> --use_retrieval --use_cache
```