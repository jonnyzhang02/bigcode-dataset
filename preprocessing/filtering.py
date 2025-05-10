"""Optimized filter script combining basic, comments, and stars filters into one map function
with streaming processing that writes filtered data directly to disk."""

import logging
import time
from functools import partial
import numpy as np
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info
from transformers import HfArgumentParser
from arguments import FilteringArguments
from utils.text_extraction import get_nl_ratio
from utils.manual_sharding import save_manual_shards

def parse_args():
    parser = HfArgumentParser(FilteringArguments)
    return parser.parse_args()

def combined_filter_map(example, args):
    """Combined filter function that handles basic, comments, and stars filters in one pass"""
    # Initialize as valid example
    is_valid = True
    stats = {}
    
    # --- Basic filters ---
    if example["max_line_length"] > args.line_max:
        is_valid = False
        stats["basic_rejected_max_line"] = 1
    elif example["avg_line_length"] > args.line_mean:
        is_valid = False
        stats["basic_rejected_avg_line"] = 1
    elif example["alphanum_fraction"] < args.alpha_frac:
        is_valid = False
        stats["basic_rejected_alphanum"] = 1
    
    # --- Stars filter ---
    # Convert None stars to 0
    stars = example.get("max_stars_count", 0)
    if stars is None:
        stars = 0
    stats["stars"] = stars
    
    if stars <= args.threshold_stars:
        is_valid = False
        stats["stars_rejected"] = 1
    
    # --- Comments filter ---
    # Calculate comment-to-code ratio
    nl_ratio = get_nl_ratio(example["content"], example["lang"].lower())
    stats["nl_ratio"] = nl_ratio
    
    if nl_ratio <= args.min_threshold_comments or nl_ratio >= args.max_threshold_comments:
        is_valid = False
        stats["comments_rejected"] = 1
    
    # Add filtering decision to stats
    stats["accepted"] = 1 if is_valid else 0
    
    # Combine the original example with the additional stats
    return {**example, **stats}

def filter_dataset_with_stats(dataset, args):
    """Apply combined filtering and collect statistics"""
    # Start timing
    t_start = time.time()
    
    # Initial dataset size
    initial_size = len(dataset)
    initial_size_gb = sum(dataset["size"]) / 1e9
    
    logger.info(f"=== Applying combined filters (basic, stars, comments) ===")
    logger.info(f"Parameters:")
    logger.info(f"- Basic: line_max={args.line_max}, line_mean={args.line_mean}, alpha_frac={args.alpha_frac}")
    logger.info(f"- Stars: threshold={args.threshold_stars}")
    logger.info(f"- Comments: min={args.min_threshold_comments}, max={args.max_threshold_comments}")
    
    # Apply combined mapping to add stats
    dataset = dataset.map(
        partial(combined_filter_map, args=args),
        num_proc=args.num_workers
    )
    
    # Keep only accepted examples
    filtered_dataset = dataset.filter(lambda x: x["accepted"] == 1)
    
    # Calculate statistics
    final_size = len(filtered_dataset)
    final_size_gb = sum(filtered_dataset["size"]) / 1e9
    rejected_pct = (initial_size - final_size) * 100 / initial_size
    size_reduction_pct = (initial_size_gb - final_size_gb) * 100 / initial_size_gb
    
    # Log statistics
    logger.info(f"Filtering completed in {time.time() - t_start:.2f} seconds")
    logger.info(f"Dataset size before filtering: {initial_size} examples, {initial_size_gb:.2f} GB")
    logger.info(f"Dataset size after filtering: {final_size} examples, {final_size_gb:.2f} GB")
    logger.info(f"Percentage of removed files: {rejected_pct:.2f}%")
    logger.info(f"Percentage of volume removed: {size_reduction_pct:.2f}%")
    
    # Calculate detailed stats if needed
    # (We could calculate more detailed stats from the dataset here)
    
    return filtered_dataset

def stream_to_disk(dataset, args):
    """Stream filtered dataset directly to disk"""
    logger.info("=== Streaming filtered dataset to disk ===")
    t_start = time.time()
    
    try:
        save_manual_shards(
            dataset, 
            user=args.hub_username, 
            remote_dataset_repo=args.remote_repo, 
            out_path=args.out_path,
            subset=args.subset
        )
        logger.info(f"Dataset successfully saved at {args.out_path}/{args.subset} in {time.time() - t_start:.2f} seconds")
    except FileExistsError:
        logger.warning(f"Output dir already exists at {args.out_path}/{args.subset}. Will not save filtered data")

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    set_verbosity_info()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    logger.info(f"** Job started with arguments: **\n{args}\n ****")
    
    # Load dataset
    t_start = time.time()
    logger.info(f"=== Loading {args.dataset_name} and subset {args.subset} ===")
    dataset = load_dataset(
        args.dataset_name, 
        split=args.split, 
        data_dir=args.subset, 
        use_auth_token=True, 
        num_proc=args.num_workers
    )
    logger.info(f"Dataset loaded in {time.time() - t_start:.2f} seconds")
    logger.info(f"Dataset: {dataset}")
    
    # Add size column if not present
    if "size" not in dataset.column_names:
        logger.info("Adding text size column")
        dataset = dataset.map(lambda example: {"size": len(example["content"])})
    
    # Apply combined filtering
    filtered_dataset = filter_dataset_with_stats(dataset, args)
    
    # Stream to disk
    stream_to_disk(filtered_dataset, args)
    
    # Log completion
    logger.info("=== Processing completed successfully ===")