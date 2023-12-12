# Retrieval Augmented Generation
The RAG experiment

**Prerequisites**
 - Install requirements: `pip install -r requirements.txt`
 - In `model_context.py` (for RAG) or `(some/all)_evaluation.py` (for baseline), replace KEY with AnyScale API key

**Data downloading / generation / preprocessing**
 - See the top-level README in `llm-logic-experiments/` for dataset details. 
 - Load embedding DB from filtered data: `python data_indexer.py`

**Basic demo**
 - Demo system: `python rag_exemplar_demo.py`

**Generating and evaluating system output**
 - Generate test outputs and run in-place correctness evaluation:
   - Performance on universal quantifier: `python all_evaluation.py`
   - Performance on existential quantifier: `python some_evaluation.py`
 - Check correctness post-generation: `python file_counter.py <file name>`
 - Adjust model parameters in `model_context.py` and prompting strategy in `templates.py`
 - For human evaluation of factuality/justification:
   - Instructions in `annotation_instructions.md`
   - For post-annotation result-checking: `python exp_counter.py <file name>`

**Baselines**
  - Baselines are generated and evaluated using the exact same scripts as RAG
  - In `(some/all)_evaluation.py`, simply set `rag=False` at the top of the file
  - Can also toggle between direct yes/no and reformatted true/false prompts with `yn=True/False`
