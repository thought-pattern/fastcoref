"""Utilities for training."""

from json import dump as json_dump
from logging import getLogger as logging_getLogger
from os import path as os_path

from torch import cuda as torch_cuda
from torch import nn as torch_nn
from torch import no_grad as torch_no_grad
from torch import tensor as torch_tensor
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utilities.util import save_all, softXEnt
from wandb import log as wandb_log
from wandb import run as wandb_run

logger = logging_getLogger(__name__)


def train(
    args,
    student_train_batches,
    teacher_train_batches,
    student,
    teacher,
    tokenizer,
    evaluator,
):
    """Train the model"""
    assert len(student_train_batches) == len(teacher_train_batches), "student and teacher batches size are different"
    t_total = len(student_train_batches) * args.train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    head_params = ["coref", "mention", "antecedent"]

    model_decay = [
        p for n, p in student.named_parameters() if not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
    ]
    model_no_decay = [
        p for n, p in student.named_parameters() if not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
    ]
    head_decay = [
        p for n, p in student.named_parameters() if any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
    ]
    head_no_decay = [
        p for n, p in student.named_parameters() if any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
    ]

    head_learning_rate = args.head_learning_rate if args.head_learning_rate else args.learning_rate
    optimizer_grouped_parameters = [
        {
            "params": model_decay,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        {"params": model_no_decay, "lr": args.learning_rate, "weight_decay": 0.0},
        {
            "params": head_decay,
            "lr": head_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {"params": head_no_decay, "lr": head_learning_rate, "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total * 0.1, num_training_steps=t_total)

    # using mixed precision
    scaler = torch_cuda.amp.GradScaler()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.train_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, tr_loss, logging_loss = 0, 0.0, 0.0
    best_f1, best_global_step = -1, -1

    torch_nn.MSELoss()
    torch_nn.CosineEmbeddingLoss()

    train_iterator = tqdm(range(int(args.train_epochs)), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(
            zip(student_train_batches, teacher_train_batches, strict=False),
            desc="Iteration",
            total=len(student_train_batches),
        )
        for _, (student_batch, teacher_batch) in enumerate(epoch_iterator):
            assert all(
                [
                    x == y
                    for x, y in zip(
                        student_batch.get("doc_key", ""),
                        teacher_batch.get("doc_key", ""),
                        strict=False,
                    )
                ]
            ), "different doc keys in the student-teacher batches"

            teacher_batch["input_ids"] = torch_tensor(teacher_batch.get("input_ids", []), device=args.device)
            teacher_batch["attention_mask"] = torch_tensor(teacher_batch.get("attention_mask", False), device=args.device)

            student_batch["input_ids"] = torch_tensor(student_batch.get("input_ids", []), device=args.device)
            student_batch["attention_mask"] = torch_tensor(student_batch.get("attention_mask", False), device=args.device)
            if "leftovers" in student_batch:
                student_batch.get("leftovers", {})["input_ids"] = torch_tensor(
                    student_batch.get("leftovers", {}).get("input_ids", []),
                    device=args.device,
                )
                student_batch.get("leftovers", {})["attention_mask"] = torch_tensor(
                    student_batch.get("leftovers", {}).get("attention_mask", False),
                    device=args.device,
                )

            student.zero_grad()
            student.train()
            teacher.eval()

            with torch_no_grad():
                outputs = teacher(teacher_batch, return_all_outputs=True)
                teacher_logits, topk_1d_indices = outputs[-2], outputs[-1]

            with torch_cuda.amp.autocast():
                outputs = student(
                    student_batch,
                    topk_1d_indices=topk_1d_indices,
                    return_all_outputs=True,
                )
                student_logits, span_mask = outputs[-1], outputs[0]

            loss = softXEnt(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                span_mask=span_mask,
            )

            tr_loss += loss.item()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scheduler.step()  # Update learning rate schedule
            scaler.update()  # Updates the scale for next iteration
            global_step += 1

            # Log metrics
            if global_step % args.logging_steps == 0:
                loss = (tr_loss - logging_loss) / args.logging_steps
                logger.info(f"\nloss step {global_step}: {loss}")
                wandb_log({"loss": loss}, step=global_step)
                logging_loss = tr_loss

            # Evaluation
            if global_step % args.eval_steps == 0:
                results = evaluator.evaluate(student, prefix=f"step_{global_step}")
                wandb_log(results, step=global_step)

                f1 = results.get("f1", False)
                if f1 > best_f1:
                    best_f1, best_global_step = f1, global_step
                    wandb_run.summary["best_f1"] = best_f1

                    # Save model
                    output_dir = os_path.join(args.output_dir, "model")
                    save_all(tokenizer=tokenizer, model=student, output_dir=output_dir)
                logger.info(f"best f1 is {best_f1} on global step {best_global_step}")

    with open(os_path.join(args.output_dir, "best_f1.json"), "w") as f:
        json_dump({"best_f1": best_f1, "best_global_step": best_global_step}, f)

    computed_return_value = global_step, tr_loss / global_step
    return computed_return_value
