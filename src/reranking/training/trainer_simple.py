import logging
from contextlib import nullcontext
from itertools import islice

import torch

import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader

from reranking.training.training_configs import get_config_from_name

# Settings ===========================================


_logger = logging.getLogger("reranking.train.trainer")


if torch.cuda.is_available():
    _logger.debug("Running on CUDA.")
    device = torch.device("cuda")
else:
    _logger.debug("CUDA is not available.")
    device = torch.device("cpu")


SEED = 0
torch.manual_seed(SEED)


def train(
    config_name: str,
    gradient_clip: float = 1.0,
):
    _logger.info("Loading training configuration")
    training_config = get_config_from_name(config_name)
    _logger.info(f"Training configuration loaded: {training_config.config_name}")

    model = training_config.model
    dataset = training_config.dataset
    optimizer = training_config.optimizer

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        pin_memory=True,
        num_workers=1,
    )

    num_validation_batches = training_config.validation_size // training_config.batch_size
    validation_batches = list(islice(iter(dataloader), num_validation_batches))
    val_dataloader = validation_batches

    model.to(device)
    model = torch.compile(model)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    def step(current_loss):
        if scaler is not None:
            scaler.scale(current_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            current_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        optimizer.zero_grad()

    global_step = 0

    for epoch in range(training_config.epochs):
        _logger.info(f"Starting epoch {epoch + 1}/{training_config.epochs}")

        for links, entities, labels, qids in dataloader:
            global_step += 1
            links = links.to(device, non_blocking=True)
            entities = entities.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # qids = qids.to(device, non_blocking=True)

            batch_data = {
                "mention_tokens": links,
                "entity_tokens": entities,
                "labels": labels,
                "qids": qids,
            }

            autocast_context = torch.autocast(device_type="cuda") if use_amp else nullcontext()
            with autocast_context:
                loss = model.train_step(batch_data)
            step(loss)
            if global_step % training_config.save_each == 0:
                path = training_config.get_output_path(global_step)
                _logger.info(f"Saving model at step {global_step} to {path}")
                model.save(path)
            if global_step % 500 == 0:
                wandb.log({"train/loss": loss.item()}, step=global_step)
                _logger.info(f"Step {global_step}, loss: {loss.item():.4f}")
            if global_step % training_config.validate_each == 0:
                model.eval()

                correct = 0
                total = 0

                total_loss = 0.0
                val_steps = 0
                for links, entities, labels, qids in val_dataloader:
                    links = links.to(device, non_blocking=True)
                    entities = entities.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    # qids = qids.to(device, non_blocking=True)

                    print(links.device, entities.device, labels.device)

                    val_steps += 1

                    with torch.inference_mode():
                        if (
                            training_config.config_name == "pairwise_mlp"
                            or training_config.config_name == "pairwise_mlp_debug"
                            or training_config.config_name == "full_lealla"
                            or training_config.config_name == "full_lealla_r"
                            or training_config.config_name == "full_lealla_r_192"
                            or training_config.config_name == "full_lealla_192"
                        ):
                            probs = model.score_from_tokens(links, entities)
                        elif training_config.config_name == "fusion":
                            probs = model.score_from_tokens_and_qids(links, entities, qids)
                        else:
                            probs = model.score_from_tokens(links, qids)
                        loss = torch.nn.functional.binary_cross_entropy(probs, labels.float())
                        total_loss += loss.item()
                        predictions = (probs > 0.5).long()
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                val_loss = total_loss / max(val_steps, 1)
                accuracy = correct / max(total, 1)
                _logger.info(f"Validation loss: {val_loss:.4f}")
                _logger.info(f"Validation accuracy: {accuracy:.4f}")
                wandb.log(
                    {
                        "validation/loss": val_loss,
                        "validation/accuracy": accuracy,
                    },
                    step=global_step,
                )

                model.train()

        _logger.info(f"Epoch {epoch + 1} finished.")
    model.save(training_config.get_output_path(global_step))


# Training ===========================================
def train_ddp(config_name: str, master_port: str = "12355"):
    _logger.warning(
        "train_ddp is deprecated and now runs single-device training. master_port is ignored."
    )
    train(config_name)


if __name__ == "__main__":
    train("pairwise_mlp")
