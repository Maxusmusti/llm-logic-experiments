# Retrieval Augmented Generation
The RAG experiment

Usage:
 - Test system: `python exemplar_test.py`
 - Run evaluation:
   - Performance on universal quantifier: `python all_evaluation.py`
   - Performance on existential quantifier: `python some_evaluation.py`
 - Load new embedding DB from data: `python data_indexer.py`
 - Adjust model parameters in `model_context.py` and prompting strategy in `templates.py`
 - NOTE: In `model_context.py` (for RAG) or `(some/all)_evaluation.py` (for baseline), replace KEY with AnyScale key
