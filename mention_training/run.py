# import sys
# from pathlib import Path
#
# # setting parent path
# sys.path.append(str(Path(__file__).parent.parent))

"""Utilities for run."""

from logging import INFO as logging_INFO
from logging import basicConfig as logging_basicConfig
from logging import getLogger as logging_getLogger
from os import mkdir as os_mkdir
from os import path as os_path
from shutil import rmtree as shutil_rmtree
from sys import path as sys_path

from models.mention_modeling import FastMention
from torch import cuda as torch_cuda
from torch import device as torch_device
from training import train
from transformers import AutoConfig, AutoTokenizer
from utilities import coref_dataset
from utilities.cli import parse_args
from utilities.collate import DynamicBatchSampler, LeftOversCollator
from utilities.consts import SUPPORTED_MODELS
from utilities.eval_mention import Evaluator
from utilities.util import set_seed
from wandb import init as wandb_init

sys_path.append("/home/nlp/shon711/fast-coref")

# Setup logging
logger = logging_getLogger(__name__)
logging_basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging_INFO,
)


def main():
    args = parse_args()

    if args.experiment_name is not None:
        wandb_init(project=args.experiment_name, config=args)

    if args.output_dir is not None:
        if os_path.exists(args.output_dir):
            if args.overwrite_output_dir:
                shutil_rmtree(args.output_dir)
                logger.info(f"--overwrite_output_dir used. directory {args.output_dir} deleted!")
            else:
                raise ValueError(f"Output directory ({args.output_dir}) already exists. Use --overwrite_output_dir to overcome.")
        os_mkdir(args.output_dir)
    else:
        if args.do_train:
            raise ValueError("Output directory is required while do_train=True.")
        else:
            if args.output_file is None:
                raise ValueError("Output directory or output file is required.")

    # Setup CUDA, GPU & distributed training
    device = torch_device(args.device if torch_cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = 1
    set_seed(args)

    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        add_prefix_space=True,
        cache_dir=args.cache_dir,
    )

    model, loading_info = FastMention.from_pretrained(
        args.model_name_or_path,
        output_loading_info=True,
        config=config,
        cache_dir=args.cache_dir,
        args=args,
    )

    if model.base_model_prefix not in SUPPORTED_MODELS:
        raise NotImplementedError(f"Model not supporting {args.model_type}, choose one of {SUPPORTED_MODELS}")
    args.base_model = model.base_model_prefix

    model.to(args.device)
    for key, val in loading_info.items():
        logger.info(f"{key}: {val}")

    t_params, h_params = [p / 1000000 for p in model.num_parameters()]
    logger.info(f"Parameters: {t_params + h_params:.1f}M, Transformer: {t_params:.1f}M, Head: {h_params:.1f}M")

    # load datasets
    dataset, dataset_files = coref_dataset.create(
        tokenizer=tokenizer,
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        cache_dir=args.cache_dir,
    )
    args.dataset_files = dataset_files

    collator = LeftOversCollator(tokenizer=tokenizer, device=args.device, max_segment_len=args.max_segment_len)
    eval_dataloader = DynamicBatchSampler(
        dataset[args.eval_split],
        collator=collator,
        max_tokens=args.max_tokens_in_batch,
        max_segment_len=args.max_segment_len,
    )
    evaluator = Evaluator(args=args, eval_dataloader=eval_dataloader)

    # Training
    if args.do_train:
        train_sampler = DynamicBatchSampler(
            dataset.get("train", False),
            collator=collator,
            max_tokens=args.max_tokens_in_batch,
            max_segment_len=args.max_segment_len,
        )
        train_batches = coref_dataset.create_batches(
            sampler=train_sampler,
            dataset_files=args.dataset_files,
            cache_dir=args.cache_dir,
        ).shuffle(seed=args.seed)
        logger.info(train_batches)

        global_step, tr_loss = train(args, train_batches, model, tokenizer, evaluator)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

    # Evaluation
    results = evaluator.evaluate(model)

    # model.push_to_hub("lingmess-coref", organization='biu-nlp', use_temp_dir=True)
    # tokenizer.push_to_hub("lingmess-coref", organization='biu-nlp', use_temp_dir=True)

    return results


if __name__ == "__main__":
    main()
