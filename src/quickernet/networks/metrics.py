from ..datasets.dataset import Dataset
from .graph import DirectedGraphModel
from ..nodes.utils import debinarize


def get_accuracy(model: DirectedGraphModel, dataset: Dataset):
    def multiclass_precision_and_recall(correct_preds, all_predictions, target_class, labels):
        # failed because all_predictions is a single-element array
        predictions_in_class = all_predictions == target_class
        true_pos = correct_preds[predictions_in_class].sum().item()
        false_pos = (sum(predictions_in_class) - true_pos).item()
        false_neg = sum((~predictions_in_class)[
                        labels == target_class]).item()
        prec_denom = true_pos + false_pos
        rec_denom = true_pos + false_neg
        precision = 0 if prec_denom == 0 else true_pos / prec_denom
        recall = 0 if rec_denom == 0 else true_pos / rec_denom
        return precision, recall

    dataset.debinarize()
    num_classes = dataset.num_classes()
    expected_outputs = dataset._labels
    forward_output = model.forward(dataset._inputs)
    outputs = debinarize(forward_output)
    correct_preds = outputs == expected_outputs
    accuracies = {'all': (sum(correct_preds) / len(outputs)).item()}
    accuracies.update({f"class_{i}": (multiclass_precision_and_recall(
        correct_preds, outputs, i, dataset._labels)) for i in range(num_classes)})
    dataset.binarize(num_classes)
    return accuracies
