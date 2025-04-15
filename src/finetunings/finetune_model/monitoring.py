from typing import Any
import torch

from utils.running_averages import RunningAverages


def _get_wandb_logs(
    loss_item: float,
    r_at_1: float,
    r_at_10: float,
    running_averages: RunningAverages,
    **kwargs
) -> dict:
    logs = {
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
    logs.update(kwargs)
    return logs


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


def get_gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float | None:
    """Calculates the gradient norm of a model.

    Args:
        model (torch.nn.Module): The model whose gradient norm is to be calculated.
        norm_type (float, optional): The type of norm to calculate. Defaults to 2.0.

    Returns:
        float: The calculated gradient norm or None if no gradients are found in the model params.
    """
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    if len(grads) == 0:
        return None
    norm = torch.cat(grads).norm(norm_type)
    return norm.item()


def _calculate_recalls(outputs, labels):
    r_at_1 = batch_recall(outputs, labels, k=1)
    r_at_10 = batch_recall(outputs, labels, k=10)
    return r_at_1, r_at_10


def _update_metrics(running_averages, loss_item, r_at_1, r_at_10):
    running_averages.update_loss(loss_item)
    running_averages.update_recall(r_at_1, r_at_10)


def _log_metrics_to_wandb(
    loss_item, r_at_1, r_at_10, running_averages, additional_metrics={}
):
    import wandb

    wand_dict = _get_wandb_logs(
        loss_item, r_at_1, r_at_10, running_averages, **additional_metrics
    )
    wandb.log(wand_dict)


def process_metrics(
    outputs: torch.tensor,
    labels: torch.tensor,
    loss_item: float,
    running_averages: RunningAverages,
    additional_metrics: dict[str, Any] = {},
):
    r_at_1, r_at_10 = _calculate_recalls(outputs, labels)
    _update_metrics(running_averages, loss_item, r_at_1, r_at_10)
    _log_metrics_to_wandb(
        loss_item, r_at_1, r_at_10, running_averages, additional_metrics
    )
