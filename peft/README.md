# Parameter-Efficient Fine-Tuning
The PEFT experiment

**Main usage:**
  - Install requirements: `pip3 install -r requirements.txt`
  - Demo script: `accelerate launch --config_file ds_zero3_cpu.yaml demo/demo.py`

**Data downloading / generation / preprocessing**
  - See the top-level README in `llm-logic-experiments/` for dataset details. 
  - There is a script to get dataset summary statistics and that can be run with `python3 util/dataset_analysis.py`

**Training your baselines**
  - See README in `llm-logic-experiments/rag/` for baseline details. 
 
**Training your experiments**
  - Training a new model or testing an existing model is done with `accelerate launch --config_file ds_zero3_cpu.yaml parameter_efficient_fine_tuning.py`. The script defines a PEFT configuration, loads the dataset, splits it into train and test datasets, preprocesses them with a tokenizer, initializes the optimizer and learning rate scheduler, trains a model which is saved into a user-specified directory or loads a model from an existing directory, and finally evaluates the model on the test split. The descriptions of each model iteration are in `results/model_descriptions`. The system uses the `ds_zero3_cpu.yaml` file to configure DeepSpeed which is used to handle FP16 mixed precision. 

**Evaluating your model output**
  - For evaluating the PEFT system, we measured the system on 2 fronts: correctness and explanation quality. For correctness, we determine that the system is correct if and only if it responds with the correct true or false response. For explanation quality, we measure factuality and justification quality. Factuality is whether the facts and information presented in the explanation are verifiably true, or if they are incorrect/hallucinated. Justification is whether the explanation validly leads to the same conclusion as provided in the answer. The actual model outputs are under the `results/model_outputs` directory. Using the txt files in this directory, you can run the accuracy evaluation script with `python3 util/accuracy_evaluation.py` and it will print out the correctness metric for each of the models. Next, the human evaluation results for the models are in the `results/human_evaluation/` directory and to get the explanation evaluation metrics, you can run the explanation evaluation script with `python3 util/explanation_evaluation.py`
 