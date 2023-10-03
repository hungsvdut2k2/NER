import evaluate
from seqeval.metrics import classification_report
import numpy as np

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    y_true = labels.tolist()
    y_pred = predictions.tolist()

    y_true = [["B-" + str(label)] for label in y_true]
    y_pred = [["B-" + str(label)] for label in y_pred]
    metrics = classification_report(y_true, y_pred, digits=3)
    print(metrics)

    return clf_metrics.compute(predictions=predictions, references=labels)
