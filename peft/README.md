# Parameter-Efficient Fine-Tuning
The PEFT experiment

Usage:
 - Install requirements: `pip3 install -r requirements.txt`
 - Train or test model: `accelerate launch --config_file ds_zero3_cpu.yaml parameter_efficient_fine_tuning.py`
 - Run accuracy evaluation script: `python3 util/results_analysis.py`


