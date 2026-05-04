# Ewen

This repository contains the upcoming anonymous code release for **Bisat**, an EEG foundation model that connects language reasoning with EEG-conditioned modeling.

The release will include the core implementation needed to reproduce the main training and evaluation pipeline:

- `model/`: Bisat model components, including the Qwen-based backbone wrapper, EEG tokenizer modules, channel layout, covariate conditioning, instruction heads, and open-ended generation interface.
- `dataset.py`, `downstream_dataset.py`, `multitask_h5.py`: dataset definitions and H5-based multitask EEG loading utilities.
- `utils.py`: shared utilities for tokenizers, metrics, seeding, schedulers, checkpoint loading, and training support.
- `train_pretrain.py`: pretraining entry point for EEG-token and text-conditioned objectives.
- `train_instruction_bisat_qwen.py`: shared instruction tuning implementation for closed-set EEG tasks and EEG-conditioned language modeling.
- `train_instruction_open_ended.py`: open-ended instruction tuning entry point. 
- `run_train_pretrain_bisat_qwen.sh`, `run_train_instruction_bisat_qwen.sh`, `run_train_instruction_open_ended.sh`: example launch scripts.
- `metrics/`: evaluation metrics used by the training and validation scripts.

Open-ended EEG generation is supported through `train_instruction_open_ended.py` and `BisatModel.generate_text_from_eeg(...)`. Classifier-style instruction heads are also included for downstream benchmark evaluation.
