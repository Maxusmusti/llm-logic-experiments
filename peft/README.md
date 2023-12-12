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


