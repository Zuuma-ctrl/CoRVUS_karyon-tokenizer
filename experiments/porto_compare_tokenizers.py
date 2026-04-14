"""
CLI experiment: compare numeric tokenizers on Porto Seguro dataset.

Usage examples
--------------
python experiments/porto_compare_tokenizers.py --mode ordered --epochs 10
python experiments/porto_compare_tokenizers.py --mode linear  --epochs 10
python experiments/porto_compare_tokenizers.py --mode center  --epochs 10 --num_centers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.data.porto import load_porto_data
from src.models.mixed_tabular_model import MixedTabularModel
from src.training.metrics import compute_metrics
from src.training.train_eval import evaluate, get_aux, set_seed, train_one_epoch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare numeric tokenizers on Porto Seguro data."
    )
    parser.add_argument(
        "--mode",
        choices=["linear", "center", "ordered"],
        default="ordered",
        help="Numeric tokenizer mode (default: ordered)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_token", type=int, default=32, help="Token dimension")
    parser.add_argument(
        "--num_centers", type=int, default=8, help="Centers (center mode)"
    )
    parser.add_argument(
        "--num_thresholds", type=int, default=8, help="Thresholds (ordered mode)"
    )
    parser.add_argument(
        "--mlp_hidden",
        nargs="+",
        type=int,
        default=[256, 128],
        help="MLP hidden layer sizes (default: 256 128)",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="MLP dropout")
    parser.add_argument(
        "--valid_size", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/porto-seguro-safe-driver-prediction",
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save results (default: outputs/)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode} | Seed: {args.seed} | Epochs: {args.epochs}")

    # ---- Data ----
    print(f"\nLoading data from: {args.data_dir}")
    prep, train_ds, valid_ds = load_porto_data(
        args.data_dir,
        valid_size=args.valid_size,
        random_state=args.seed,
    )
    print(
        f"  Numeric: {prep.num_numeric}  Categorical: {prep.num_cat}  Binary: {prep.num_bin}"
    )
    print(f"  Train: {len(train_ds)}  Valid: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ---- Model ----
    model = MixedTabularModel(
        num_numeric=prep.num_numeric,
        cat_vocab_sizes=prep.cat_vocab_sizes(),
        bin_vocab_sizes=prep.bin_vocab_sizes(),
        d_token=args.d_token,
        mlp_hidden=args.mlp_hidden,
        mode=args.mode,
        num_centers=args.num_centers,
        num_thresholds=args.num_thresholds,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Training loop ----
    print("\nTraining...")
    best_auc = 0.0
    history = []
    y_valid = valid_ds.tensors["y"].numpy()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        preds = evaluate(model, valid_loader, device)
        metrics = compute_metrics(y_valid, preds)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        best_auc = max(best_auc, metrics["auc"])
        print(
            f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
            f"AUC={metrics['auc']:.4f} | LogLoss={metrics['logloss']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f}"
        )

    print(f"\nBest AUC: {best_auc:.4f}")

    # ---- Save results ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_seed{args.seed}"

    results_path = output_dir / f"results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "seed": args.seed,
                "best_auc": best_auc,
                "final_metrics": history[-1],
                "history": history,
                "args": vars(args),
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {results_path}")

    # ---- Ordered: save thresholds and assignment summary ----
    if args.mode == "ordered":
        print("\nCollecting ordered-threshold internals...")
        aux_data = get_aux(model, valid_loader, device)

        # thresholds per feature
        thresholds = model.numeric_tokenizer.get_thresholds().detach().cpu().numpy()
        thresh_path = output_dir / f"thresholds_{tag}.npy"
        np.save(thresh_path, thresholds)
        print(f"  Thresholds saved to: {thresh_path}  shape={thresholds.shape}")

        if "assign" in aux_data:
            assign_all = np.concatenate(aux_data["assign"], axis=0)  # (N, M, K)
            assign_mean = assign_all.mean(axis=0)                      # (M, K)
            assign_path = output_dir / f"assign_mean_{tag}.npy"
            np.save(assign_path, assign_mean)
            print(f"  Assignment mean saved to: {assign_path}  shape={assign_mean.shape}")

            print("\n  Top-5 numeric features by assignment entropy:")
            entropy = -(assign_mean * np.log(assign_mean + 1e-9)).sum(axis=1)
            top5 = np.argsort(entropy)[::-1][:5]
            for rank, feat_idx in enumerate(top5):
                col_name = prep.numeric_cols[feat_idx]
                print(f"    {rank+1}. {col_name}  entropy={entropy[feat_idx]:.3f}")


if __name__ == "__main__":
    main()
