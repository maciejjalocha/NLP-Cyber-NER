# NER for cybersecurity

## Paper

See [Paper](https://arxiv.org/abs/2507.13870)

## Replicate

Experiments were performed using conda on Linux with Intel CPU. In order to replicate the experiments run:

```bash
# ensure you have installed conda, initialized it and sourced the bashrc.
conda env create -n nlp-cyber-ner -f envs/prod_environment.yml
conda activate nlp-cyber-ner

# For BERT Baseline and LST-NER models (separate environment)
conda env create -f envs/bert_lstner_requirements.yaml
conda activate nlp-cyber-ner-bert-lstner
```

All experiments had seed set.

Experiments and associated commits are in our online MLFlow server: [MLFlow Server (DAGsHub)](https://dagshub.com/PLtier/NLP-Cyber-NER/experiments).

### NLP-Cyber-NER Experiments

This project provides scripts to execute a variety of NER experiments for cybersecurity datasets. The main experiment types include:

- **Combined Dataset Model**: Trains and evaluates a model on the union of all datasets.
- **Cross Dataset Model**: Trains and evaluates models across different datasets (e.g., train on one, evaluate on another).
- **Multihead (Tokenmodel) Experiments**: Trains models with multiple heads for different datasets, supporting various architectural variants (e.g., tied/untied embeddings and LSTMs).
- **LST-NER Model**: Cross-domain NER using label structure transfer with graph neural networks and optimal transport.
- **BERT Baseline Model**: Standard RoBERTa NER fine-tuning (no graph component) for baseline comparison.

```bash
python nlp_cyber_ner/modeling/cross_dataset_model.py
python nlp_cyber_ner/modeling/combined_dataset_model.py
python nlp_cyber_ner/modeling/*tokenmodel*.py # different

# LST-NER and BERT Baseline (requires separate environment)
**Before running:** Update the dataset paths in the configuration section of each script to match your desired dataset.
python -m models.train_lst_ner_aptner   # graph model; also _dnrti, _cyner, _attackner
python -m models.BERT_APTNER            # baseline; also BERT_DNRTI, BERT_CYNER, BERT_ATTACKNER
```

### Experiment Tracking with MLflow

All experiments are automatically logged to MLflow. You can set the environment variable `MLFLOW_TRACKING_URI` to log results to an external MLflow tracking server. Otherwise, results are logged locally by default.

After running experiments, you can launch the MLflow UI on your local machine to browse results:

```bash
mlflow ui
```

This will start a web server where you can explore experiment runs, metrics, and artifacts.

### Looking up predictions
Cross-dataset models and Combined-dataset model, BERT-base-NER model, and LST-NER model output predictions into `artifacts/predictions` folder.
These can be for example compared to ground truth using our lookup tools `lookup.html` and `lookup2.html`.

## Development

Development was performed with venv, packages are in [dev_requirements.txt](envs/dev_requirements.txt).

## Cite

Please cite this work if you use it.
