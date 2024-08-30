import torch

from utils.running_averages import RunningAverages


def get_wandb_logs(
    loss_item: float, r_at_1: float, r_at_10: float, running_averages: RunningAverages
) -> dict:
    return {
        "loss": loss_item,
        "r_at_1": r_at_1,
        "r_at_10": r_at_10,
        "running_loss": running_averages.loss,
        "running_r_at_1": running_averages.recall_1,
        "running_r_at_10": running_averages.recall_10,
        "running_loss_big": running_averages.loss_big,
        "running_r_at_1_big": running_averages.recall_1_big,
        "running_r_at_10_big": running_averages.recall_10_big,
    }


def batch_recall(outputs: torch.tensor, target: torch.tensor, k: int = 1) -> float:
    """Calculates recall inside the batch.

    The calculation is done per each row. The exact values of outputs and target are not importand only orderings matter.
    Consequently this works both with logits and softmax. If k is greater than the number of classes **returns 0**.

    Args:
        outputs (torch.tensor): Matrix where each row corresponds to one multiclass classification.
        target (torch.tensor): Matrix where each row corresponds to one multiclass classification, same shape as outputs.
        k (int, optional): Recall at K. Defaults to 1.

    Returns:
        float: Recall at K for this batch.
    """
    if len(outputs[0]) < k:  # batch is too small.
        return 0.0
    _, top_indices = outputs.topk(k, dim=-1)
    top_values = target.gather(-1, top_indices)
    recall_per_row = top_values.any(dim=-1).float()
    return recall_per_row.mean().item()
