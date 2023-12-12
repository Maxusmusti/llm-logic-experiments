# Parameter-Efficient Fine-Tuning
The PEFT experiment

Main usage:
 - Install requirements: `pip3 install -r requirements.txt`
 - Demo script: `accelerate launch --config_file ds_zero3_cpu.yaml demo/demo.py`
 - Train a new model or test an existing model: `accelerate launch --config_file ds_zero3_cpu.yaml parameter_efficient_fine_tuning.py`

Util scripts:
 - Run accuracy evaluation script: `python3 util/accuracy_evaluation.py`
 - Run explanation evaluation script: `python3 util/explanation_evaluation.py`
 - Get dataset summary statistics: `python3 util/dataset_analysis.py`

Saved results:
 - All model outputs: `results/model_outputs/`
 - Human evaluation results: `results/human_evaluation/`
 - Model descriptions: `results/model_descriptions`

1) Data downloading / generation / preprocessing
 - See the top-level README in `llm-logic-experiments/` for dataset details. 

2) Training your baselines
 - See README in `llm-logic-experiments/rag/`. 
 
3) Training your experiments
 - 

4) Evaluating your model output
 - For evaluating the PEFT system, we measured the system on 2 fronts: correctness and explanation quality. For correctness, we determine that the system is correct if and only if it responds with the correct true or false response. For explanation quality, 
