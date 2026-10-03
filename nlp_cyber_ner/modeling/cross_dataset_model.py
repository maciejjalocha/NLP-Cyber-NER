import gc
import os

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nlp_cyber_ner.config import (
    DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    load_dotenv,
)
from nlp_cyber_ner.dataset import (
    Preprocess,
    list_to_conll,
    preds_to_tags,
    read_iob2_file,
    remove_leakage,
)
from nlp_cyber_ner.span_f1 import span_f1

MAX_LEN = 120

end_labels = {
    "B-Organization",
    "O",
    "I-Malware",
    "B-System",
    "I-Vulnerability",
    "I-Organization",
    "I-System",
    "B-Vulnerability",
    "B-Malware",
}

print("goes well")
cyner_path = PROCESSED_DATA_DIR / "cyner"
cyner_train_path = cyner_path / "train.unified"
cyner_dev_path = cyner_path / "valid.unified"
cyner_test_path = cyner_path / "test.unified"

cyner_train_data = read_iob2_file(cyner_train_path)
cyner_dev_data = read_iob2_file(cyner_dev_path)
cyner_test_data = read_iob2_file(cyner_test_path)
A = set(tag for _, tags in cyner_train_data for tag in tags)
B = set(tag for _, tags in cyner_dev_data for tag in tags)
C = set(tag for _, tags in cyner_test_data for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."
print("cyner loaded")

aptner_path = PROCESSED_DATA_DIR / "aptner"
aptner_train_path = aptner_path / "train.unified"
aptner_dev_path = aptner_path / "valid.unified"
aptner_test_path = aptner_path / "test.unified"
aptner_train_data = read_iob2_file(aptner_train_path)
aptner_dev_data = read_iob2_file(aptner_dev_path)
aptner_test_data = read_iob2_file(aptner_test_path)
A = set(tag for _, tags in aptner_train_data for tag in tags)
B = set(tag for _, tags in aptner_dev_data for tag in tags)
C = set(tag for _, tags in aptner_test_data for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."
print("aptner loaded")

attacker_path = PROCESSED_DATA_DIR / "attacker"
attacker_train_path = attacker_path / "train.unified"
attacker_dev_path = attacker_path / "valid.unified"
attacker_test_path = attacker_path / "test.unified"
attacker_train_data = read_iob2_file(attacker_train_path, word_index=0, tag_index=1)
attacker_dev_data = read_iob2_file(attacker_dev_path, word_index=0, tag_index=1)
attacker_test_data = read_iob2_file(attacker_test_path, word_index=0, tag_index=1)
A = set(tag for _, tags in attacker_train_data for tag in tags)
B = set(tag for _, tags in attacker_dev_data for tag in tags)
C = set(tag for _, tags in attacker_test_data for tag in tags)
assert A == B == C == end_labels, (
    "The labels in the train_data, dev_data and test_data sets are not the same."
)
print("attacker loaded")

dnrti_path = PROCESSED_DATA_DIR / "dnrti"
dnrti_train_path = dnrti_path / "train.unified"
dnrti_dev_path = dnrti_path / "valid.unified"
dnrti_test_path = dnrti_path / "test.unified"

dnrti_train_data = read_iob2_file(dnrti_train_path, word_index=0, tag_index=1)
dnrti_dev_data = read_iob2_file(dnrti_dev_path, word_index=0, tag_index=1)
dnrti_test_data = read_iob2_file(dnrti_test_path, word_index=0, tag_index=1)
A = set(tag for _, tags in dnrti_train_data for tag in tags)
B = set(tag for _, tags in dnrti_dev_data for tag in tags)
C = set(tag for _, tags in dnrti_test_data for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."
print("dnrti loaded")

load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri is not None:
    print(f"MLFLOW_TRACKING_URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
print("mlflow and load dotenv loaded")

BATCH_SIZE = 32
DIM_EMBEDDING = 100
LSTM_HIDDEN = 100
LEARNING_RATE = 0.001
EPOCHS = 15
CLIPPING = 5.0
hyperparams = {
    "BATCH_SIZE": BATCH_SIZE,
    "DIM_EMBEDDING": DIM_EMBEDDING,
    "LSTM_HIDDEN": LSTM_HIDDEN,
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "CLIPPING": 5.0,
}


class TaggerModel(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super().__init__()
        self.embed = nn.Embedding(nwords, DIM_EMBEDDING)
        self.drop1 = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(DIM_EMBEDDING, LSTM_HIDDEN, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(LSTM_HIDDEN * 2, ntags)

    def forward(self, input_data):
        word_vectors = self.embed(input_data)
        regular1 = self.drop1(word_vectors)
        output, hidden = self.lstm(regular1)
        regular2 = self.drop2(output)

        predictions = self.fc(regular2)
        return predictions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def train(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    idx2word: list[str],
    idx2label: list[str],
    input_size: int,
) -> TaggerModel:
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, BATCH_SIZE)  # drop_last=True
    n_batches = len(train_loader)

    model = TaggerModel(len(idx2word), len(idx2label))
    model = model.to(device)  # run on cuda if possible
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    # creating the batches

    for epoch in range(EPOCHS):
        model.train()
        # reset the gradient
        print(f"Epoch {epoch + 1}\n-------------------------------")
        loss_sum = 0

        # loop over batches
        # types for convenience
        batch_X: torch.Tensor
        batch_y: torch.Tensor
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # run on cuda if possible

            optimizer.zero_grad()

            predicted_values = model.forward(batch_X)

            # Cross entropy request (predictions, classes) shape for predictions, and (classes) for batch_y

            # calculate loss
            loss = loss_function(
                predicted_values.view(batch_X.shape[0] * input_size, -1), batch_y.flatten()
            )  # TODO: Input
            loss_sum += loss.item()  # avg later

            # update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIPPING)
            optimizer.step()

        print(f"Average loss after epoch {epoch + 1}: {loss_sum / n_batches}")

    # set to evaluation mode
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    return model


def test_pass(
    model: TaggerModel,
    dev_X: torch.Tensor,
    return_labels_idx: bool = True,
) -> torch.Tensor:
    """
    Performs forward pass on the model and returns the predictions.
    if return_labels is True, returns the labels, instead of probabilities.
    """
    all_predictions = []

    dev_dataset = TensorDataset(dev_X)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for (batch_dev,) in dev_loader:
            batch_dev = batch_dev.to(device)  # run on cuda if possible
            batch_probas_dev = model.forward(batch_dev)
            all_predictions.append(batch_probas_dev.cpu())

    # eval using Span_F1

    # to minimise memory usage, we delete the model and dev_X, even though it takes performance hit
    del dev_X
    del model
    gc.collect()
    torch.cuda.empty_cache()

    probas_dev = torch.cat(all_predictions, dim=0)
    if not return_labels_idx:
        return probas_dev
    else:
        labels_idx_dev = torch.argmax(probas_dev, 2)
        return labels_idx_dev


def evaluate(
    gt_labels: tuple,
    pred_labels: list,
):
    metrics = span_f1(gt_labels, pred_labels)

    return metrics


train_packs = [
    ("cyner", cyner_train_data),
    ("aptner", aptner_train_data),
    ("attacker", attacker_train_data),
    ("dnrti", dnrti_train_data),
]

dev_packs = [
    ("cyner", cyner_dev_data),
    ("aptner", aptner_dev_data),
    ("attacker", attacker_dev_data),
    ("dnrti", dnrti_dev_data),
]

for train_pack_name, train_data in train_packs:
    for dev_pack_name, dev_data in dev_packs:
        train_data, _ = remove_leakage(train_data, dev_data)
        transformer = Preprocess()

        max_len = MAX_LEN
        # can use obtained from train, but in order to make models more comparable, make them the same.
        # max_len = max([len(x[0]) for x in train_data])

        # input_size = max_len
        train_X, train_y, idx2word, idx2label = transformer.build_vocab(
            train_data, len(train_data), max_len
        )

        name: str = f"train-{train_pack_name}-eval-{dev_pack_name}"
        print(f"train {name}")
        model = train(train_X, train_y, idx2word, idx2label, input_size=max_len)

        dev_X, _ = transformer.transform_prep_data(dev_data, len(dev_data), max_len)
        dev_tokens, dev_labels = list(zip(*dev_data))

        # train_X, train_y, dev_X, dev_y, test_X, test_y, idx2word, idx2label, max_len = train_pack
        mlflow.set_experiment(name)
        with mlflow.start_run(run_name=name):
            assert isinstance(dev_labels, tuple), "Dev y is not a list!"

            pred_labels_idx_dev = test_pass(model, dev_X, return_labels_idx=True)

            labels_dev = preds_to_tags(idx2label, dev_labels, pred_labels_idx_dev)

            metrics = evaluate(dev_labels, labels_dev)

            store_preds_path = MODELS_DIR / "predictions" / f"{name}.txt"
            store_trains_path = MODELS_DIR / "train" / f"{name}.txt"

            list_to_conll(dev_tokens, labels_dev, store_preds_path)  # type: ignore
            train_tokens, train_labels = list(zip(*train_data))
            list_to_conll(train_tokens, train_labels, store_trains_path)  # type: ignore

            # Log the trained model
            model_path = MODELS_DIR / f"{name}.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")

            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)
            mlflow.log_params(
                {
                    "train_size": train_X.shape[0],
                    "train_input_size": train_X.shape[1],
                    "dev_size": dev_X.shape[0],
                    "dev_input_size": max(len(x[0]) for x in dev_data),
                }
            )
            mlflow.log_artifact(str(store_preds_path), artifact_path="predictions")
            mlflow.log_artifact(str(store_trains_path), artifact_path="train")
