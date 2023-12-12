# llm-logic-experiments
Attempting to improve generics reasoning in LLMs through various experiments.

To learn more about the experiments ant their significance, read our report (`generics_rag_peft_report.pdf`).
Abstract:
```
This is an extension of "Penguins Don't Fly: Reasoning about Generics through Instantiations
and Exceptions" by Allaway et al. The initial work addressed the fact that many commonsense
knowledge bases encode some generic knowledge, but rarely enumerate exceptions. To remedy
this, a novel system was proposed to generate a dataset of generics and exemplars. Our work
takes this initial concern a step further, analyzing the current level of understanding LLMs
possess for generics and exemplars, and finding the optimal means to alleviate any lack of
understanding.

After initial experiments, it is clear that large models (like Llama-2-70B) do not have too
much issue with question answering and reasoning pertaining to generics and exemplars. But
smaller, consumable models (like Llama-2-7B) struggle immensely. There remains the question of
whether this is a content and knowledge retainment issue, or a prompting issue. To explore and
potentially remedy this, we have built two rival solutions: one utilizing Retrieval Augmented
Generation alongside the "Penguin's Don't Fly" dataset, and the other utilizing Prompt Tuning
via PEFT (Parameter-Efficient Fine-Tuning), both proving viable.
```

**Experiments**

 - See `rag` for the Retrieval Augmented Generation experiment
 - See `peft` for the Parameter-Efficient Prompt Tuning experiment

**Data downloading / generation / preprocessing**

 - The raw data was provided by Emily Allaway
   - In `neg-exemplars-raw` are all of the unfiltered negative exemplars (exceptions/counter-examples)
   - In `pos-instantiations-raw` are all of the unfiltered positive exemplars (instantiations)
 - Each of these files were then filtered for quality and reformatted in `all-exemplars-pruned`
   - `negative.csv` was generated manually be removing fields from the unfiltered negative file
   - `positive.csv` was generated by filtering positive exemplars using `pos-instantiations-raw/prune.py`
 - This filtered data was then consumed by both the RAG and PEFT Prompt Tuning solutions
 - Scripts for dataset summary statistics are detailed in `peft/README.md`

**Baselines**

 - See README in `rag` for additional baseline details

**Contribution Breakdown**

The following is a simple breakdown of file authorship. Note that this is not actually always the case, but these can be considered as "usually true" (get it, like a generic 😁)
 - Mustafa Eyceoz is responsible for files in the `rag` dir, as well as files in the top-level
 - Justin Lee is responsible for files in the `peft` dir
