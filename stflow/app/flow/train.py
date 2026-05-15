import os
import json
import wandb
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from operator import itemgetter

import torch

from stflow.utils import set_random_seed, get_current_time, merge_fold_results
from stflow.data.dataset import HESTDatasetPath, MultiHESTDataset, padding_batcher, HESTDataset
from stflow.data.normalize_utils import get_normalize_method
from stflow.model.denoiser import Denoiser
from stflow.flow.interpolant import Interpolant
from stflow.app.flow.test import test
from stflow.hest_utils.utils import save_pkl


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir):
    normalize_method = get_normalize_method(args.normalize_method)

    print("Dataset Loading")
    sample_id_paths = [
        HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        ) for sample_id in train_sample_ids
    ]
    train_dataset = MultiHESTDataset(sample_id_paths,
                                     distribution=args.patch_distribution,
                                     normalize_method=normalize_method,
                                     sample_times=args.sample_times)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=padding_batcher(),
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0 and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # using the same train sample ids for validation
    sample_id_paths = [
        HESTDatasetPath(
                name=sample_id,
                h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
                h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
                gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list
            ),
        ) for sample_id in test_sample_ids
    ]
    val_loaders = [
        torch.utils.data.DataLoader(
            HESTDataset(
                sample_id_path, distribution="constant_1.0", 
                normalize_method=normalize_method,
                sample_times=1
            ),
            batch_size=1, collate_fn=padding_batcher()
        ) for sample_id_path in sample_id_paths
    ]

    device = args.device
    model = Denoiser(args).to(device)

    # Optional torch.compile — fused kernels, removes Python overhead.
    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=args.compile_mode)

    diffusier = Interpolant(
        args.prior_sampler,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Mixed precision (bf16 is numerically safer for flow matching; no GradScaler needed).
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    amp_enabled = amp_dtype is not None and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    print("Training")
    best_pearson, best_val_dict = -1, None
    early_stop_step = 0
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)
    for epoch in epoch_iter:
        # Accumulate loss on-device to avoid a GPU->CPU sync every step.
        loss_sum = torch.zeros((), device=device)
        n_steps = 0
        model.train()

        for step, batch in enumerate(train_loader):
            batch = [x.to(device, non_blocking=True) for x in batch]
            img_features, coords, gene_exp = batch

            noisy_exp, t_steps = diffusier.corrupt_exp(gene_exp)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                pred_exp, loss = model(
                    exp=noisy_exp,
                    img_features=img_features,
                    coords=coords,
                    labels=gene_exp,
                    t_steps=t_steps,
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()

            loss_sum = loss_sum + loss.detach()
            n_steps += 1

            if args.use_wandb and (step % args.log_every == 0):
                wandb.log({f"{args.dataset}/Train/{split_id}/loss": loss.detach().float().item()})

        avg_loss = (loss_sum / max(n_steps, 1)).item()
        epoch_iter.set_description(f"epoch: {epoch}, avg_loss: {avg_loss:.3f}")

        if args.save_step > 0 and epoch % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{epoch}.pth"))

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)
            if val_perf_dict["all"]['pearson_mean'] > best_pearson:
                best_pearson = val_perf_dict["all"]['pearson_mean']
                best_val_dict = val_perf_dict
                for patch_name, dataset_res in val_perf_dict.items():
                    with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                        json.dump(dataset_res, f, sort_keys=True, indent=4)
                
                # save_pkl(os.path.join(val_save_dir, 'inference_dump.pkl'), pred_dump)
                early_stop_step = 0

            else:
                early_stop_step += 1
                if early_stop_step >= 20:
                    print("Early stopping")
                    break

            if args.use_wandb:
                for patch_name, dataset_res in val_perf_dict.items():
                    wandb.log({
                        f"{args.dataset}/Val/{split_id}/{patch_name}/pearson_mean": dataset_res['pearson_mean'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/pearson_std": dataset_res['pearson_std'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q1": dataset_res['l2_error_q1'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q2": dataset_res['l2_error_q2'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q3": dataset_res['l2_error_q3'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q1": dataset_res['r2_score_q1'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q2": dataset_res['r2_score_q2'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q3": dataset_res['r2_score_q3'],
                    })

    return best_val_dict["all"]


def run(args):
    # get train/test splits
    split_dir = os.path.join(args.source_dataroot, args.dataset, 'splits')
    splits = os.listdir(split_dir)
    all_split_results = []
    
    for i in range(len(splits) // 2):  # each split has a train and test file so we divide by 2
        print(f"Running dataset {args.dataset} split {i}")

        train_df = pd.read_csv(os.path.join(split_dir, f'train_{i}.csv'))
        test_df = pd.read_csv(os.path.join(split_dir, f'test_{i}.csv'))

        train_sample_ids = train_df['sample_id'].tolist()
        test_sample_ids = test_df['sample_id'].tolist()

        kfold_save_dir = os.path.join(args.save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        checkpoint_save_dir = os.path.join(kfold_save_dir, 'checkpoints')
        os.makedirs(checkpoint_save_dir, exist_ok=True)

        results = main(args, i, train_sample_ids, test_sample_ids, kfold_save_dir, checkpoint_save_dir)
        all_split_results.append(results)

    kfold_results = merge_fold_results(all_split_results)
    with open(os.path.join(args.save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', default=["all"], help="LUNG, READ, HCC")
    parser.add_argument('--use_wandb', default=False)
    parser.add_argument('--source_dataroot', default="/home/username/Anonymous_STFlow/dataset/")
    parser.add_argument('--embed_dataroot', type=str, default="/home/username/Anonymous_STFlow/dataset/embed_dataroot")
    parser.add_argument('--gene_list', type=str, default='var_50genes.json')
    parser.add_argument('--save_dir', type=str, default="/home/username/Anonymous_STFlow/results_dir/")
    parser.add_argument('--feature_encoder', type=str, default='uni_v1_official', help="uni_v1_official | resnet50_trunc | ciga | gigapath")
    parser.add_argument('--normalize_method', type=str, default="log1p")
    parser.add_argument('--exp_code', type=str, default="test")
    
    # training hyperparameters
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--sample_times', type=int, default=10, help='Number of times to sample patches from each image')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=-1)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor (only used when num_workers>0)')
    parser.add_argument('--amp', type=str, default='off', choices=['off', 'bf16', 'fp16'], help='Mixed precision mode')
    parser.add_argument('--use_compile', action='store_true', help='Enable torch.compile on the denoiser')
    parser.add_argument('--compile_mode', type=str, default='default', help='torch.compile mode: default | reduce-overhead | max-autotune')
    parser.add_argument('--log_every', type=int, default=10, help='wandb log every N steps (reduces GPU->CPU syncs)')
    parser.add_argument('--loss_func', type=str, default='mse', help="mse | mae | pearson")
    parser.add_argument('--patch_distribution', type=str, default='uniform')
    parser.add_argument('--n_genes', type=int, default=50)

    # flow matching hyperparameters
    parser.add_argument('--n_sample_steps', type=int, default=5)
    parser.add_argument('--prior_sampler', type=str, default="zinb", help="gaussian | uniform | zero | zinb")
    parser.add_argument('--zinb_logits', type=float, default=0.1)
    parser.add_argument('--zinb_total_count', type=float, default=1)
    parser.add_argument('--zinb_zi_logits', type=float, default=0., help="Prob for zero inflation")  # before sigmoid

    # model hyperparameters
    parser.add_argument('--backbone', type=str, default="spatial_transformer")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--pairwise_hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--n_neighbors', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=1024, help="uni:1024, ciga:512")
    parser.add_argument('--norm', type=str, default='layer', help="batch | layer")
    parser.add_argument('--activation', type=str, default='swiglu', help="relu | gelu | swiglu")
    args = parser.parse_args()

    args.feature_dim = {
        "uni_v1_official": 1024,
        "gigapath": 1536,
        "ciga": 512,
    }[args.feature_encoder]

    set_random_seed(args.seed)

    if args.exp_code is None:
        exp_code = f"{args.backbone}::{get_current_time()}"
    else:
        exp_code = args.exp_code + f"_{args.feature_encoder}" + f"_{args.backbone}::{get_current_time()}"
    save_dir = os.path.join(args.save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)
    
    if args.use_wandb:
        wandb.init(project="spatial_transcriptomics", name=exp_code)
        wandb.config.update(args)

    print(f"Save dir: {save_dir}")
    print(args)

    if args.datasets[0] == "all":
        args.datasets = ["LUNG", "HCC", "COAD", "SKCM", "PAAD", "READ", "LYMPH_IDC", "PRAD", "IDC", "CCRCC"]

    for dataset in args.datasets:
        args.dataset = dataset
        args.save_dir = os.path.join(save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)

        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        run(args)

    if args.use_wandb:
        wandb.finish()
