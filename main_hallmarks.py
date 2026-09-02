"""
Main training script for MOTCat with Hallmarks dataset.

Key changes from original main.py:
- Uses TITAN slide-level features (768-dim) from titan_features/TCGA_TITAN_features.pkl
- Uses 50 Hallmark pathway signatures
- Uses new split format: train.csv and test.csv in separate files
- No validation during training - only evaluate on test set at final epoch
- Uses dss_survival_days and dss_censorship columns (Disease-Specific Survival)

Usage:
    python main_hallmarks.py                    # Run all 6 cancer types, 5-fold each
    python main_hallmarks.py --cancer_type BLCA # Run single cancer type
"""
from __future__ import print_function
import argparse
import os
import sys

from timeit import default_timer as timer
import numpy as np
import torch

from dataset.dataset_survival import (
    Generic_MIL_Survival_Dataset_Hallmarks,
    Generic_MIL_Survival_Split_Hallmarks
)
from utils.file_utils import save_pkl
from utils.utils import get_split_loader, get_optim


### Training settings
parser = argparse.ArgumentParser(
    description='MOTCat Survival Analysis with Hallmarks Dataset')

### Data and Split Parameters
parser.add_argument('--data_dir', type=str, default='./data',
                    help='Base data directory containing titan_features/, data_csvs/, splits/')
parser.add_argument('--cancer_type', type=str, default='ALL',
                    help='Cancer type: BLCA, BRCA, KIRC, LUAD, STAD, COADREAD, or ALL (runs all 6)')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5,
                    help='Number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1,
                    help='Start fold (Default: -1, first fold)')
parser.add_argument('--k_end', type=int, default=-1,
                    help='End fold (Default: -1, last fold)')
parser.add_argument('--results_dir', type=str, default='./results_hallmarks',
                    help='Results directory (Default: ./results_hallmarks)')
parser.add_argument('--log_data', action='store_true',
                    help='Log data using tensorboard')
parser.add_argument('--overwrite', action='store_true', default=True,
                    help='Whether to overwrite experiments (default: True)')

### Model Parameters
parser.add_argument('--model_type', type=str, choices=['motcat', 'mcat'],
                    default='motcat', help='Type of model (Default: motcat)')
parser.add_argument('--fusion', type=str, choices=['None', 'concat'],
                    default='concat', help='Type of fusion (Default: concat)')
parser.add_argument('--drop_out', action='store_true', default=True,
                    help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi', type=str, default='small',
                    help='Network size for WSI model')
parser.add_argument('--model_size_omic', type=str, default='small',
                    help='Network size for omic model')

### Optimizer Parameters
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch Size (Default: 1)')
parser.add_argument('--gc', type=int, default=32,
                    help='Gradient Accumulation Step')
parser.add_argument('--max_epochs', type=int, default=20,
                    help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='Learning rate (default: 0.0002)')
parser.add_argument('--bag_loss', type=str, choices=['nll_surv', 'cox_surv'],
                    default='nll_surv', help='Survival loss function')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='L2-regularization weight decay')
parser.add_argument('--alpha_surv', type=float, default=0.0,
                    help='Weight for uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'],
                    default='None', help='L1 regularization type')
parser.add_argument('--lambda_reg', type=float, default=1e-4,
                    help='L1-Regularization Strength')
parser.add_argument('--weighted_sample', action='store_true', default=False,
                    help='Enable weighted sampling')

### MOTCat Parameters
parser.add_argument('--use_micro_batch', action='store_true', default=False,
                    help='Enable micro-batching for large WSI bags')
parser.add_argument('--bs_micro', type=int, default=256,
                    help='Micro-batch size (Default: 256)')
parser.add_argument('--ot_impl', type=str, default='pot-uot-l2',
                    help='OT implementation (default: pot-uot-l2)')
parser.add_argument('--ot_reg', type=float, default=0.1,
                    help='OT epsilon (default: 0.1)')
parser.add_argument('--ot_tau', type=float, default=0.5,
                    help='UOT tau (default: 0.5)')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)


def train_fold(dataset, fold, args):
    """Train a single fold"""
    print(f'\n{"="*60}')
    print(f'Training Fold {fold}')
    print(f'{"="*60}')

    # Create fold results directory
    args.writer_dir = os.path.join(args.results_dir, str(fold))
    if not os.path.isdir(args.writer_dir):
        os.mkdir(args.writer_dir)

    # Setup tensorboard writer
    writer = None
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.writer_dir, flush_secs=15)

    # Load train and test splits
    split_folder = os.path.join(
        args.data_dir, 'splits', 'survival',
        f'TCGA_{args.cancer_type}_overall_survival_k={fold}'
    )
    train_csv = os.path.join(split_folder, 'train.csv')
    test_csv = os.path.join(split_folder, 'test.csv')

    print(f"Loading train split from: {train_csv}")
    print(f"Loading test split from: {test_csv}")

    # Create train split first (computes bins from train's uncensored patients)
    train_split = Generic_MIL_Survival_Split_Hallmarks(dataset, train_csv, external_bins=None)

    # Create test split using train's bins (ensures consistent label semantics)
    # This is the correct approach: train bins applied to test
    test_split = Generic_MIL_Survival_Split_Hallmarks(dataset, test_csv, external_bins=train_split.bins)

    # Normalize genomic features using train split statistics
    print("Normalizing genomic features...")
    scalers = train_split.get_scaler()
    train_split.apply_scaler(scalers)
    test_split.apply_scaler(scalers)

    print(f"Training on {len(train_split)} patients")
    print(f"Testing on {len(test_split)} patients")

    # Setup loss function
    if args.bag_loss == 'nll_surv':
        from utils.utils import NLLSurvLoss
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        from utils.utils import CoxSurvLoss
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError(f"Loss {args.bag_loss} not implemented")

    # Setup regularization
    if args.reg_type == 'omic':
        from utils.utils import l1_reg_all
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        from utils.utils import l1_reg_modules
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    # Initialize model
    print('\nInitializing model...')
    args.n_classes = 4
    args.omic_sizes = train_split.omic_sizes
    print(f"Number of Hallmark pathways: {len(args.omic_sizes)}")
    print(f"Omic sizes (first 10): {args.omic_sizes[:10]}")

    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type == 'motcat':
        from models.model_motcat import MOTCAT_Surv
        model = MOTCAT_Surv(
            ot_reg=args.ot_reg,
            ot_tau=args.ot_tau,
            ot_impl=args.ot_impl,
            fusion=args.fusion,
            omic_sizes=args.omic_sizes,
            n_classes=args.n_classes,
            wsi_dim=768  # TITAN features are 768-dim
        )
    elif args.model_type == 'mcat':
        from models.model_coattn import MCAT_Surv
        model = MCAT_Surv(
            fusion=args.fusion,
            omic_sizes=args.omic_sizes,
            n_classes=args.n_classes,
            wsi_dim=768  # TITAN features are 768-dim
        )
    else:
        raise NotImplementedError(f"Model {args.model_type} not implemented")

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)

    print('Model initialized!')

    # Setup optimizer
    optimizer = get_optim(model, args)

    # Setup data loaders
    train_loader = get_split_loader(
        train_split, training=True, testing=False,
        weighted=args.weighted_sample, mode='coattn_hallmarks',
        batch_size=args.batch_size
    )
    test_loader = get_split_loader(
        test_split, training=False, testing=False,
        mode='coattn_hallmarks', batch_size=args.batch_size
    )

    # Import trainer functions
    from trainer.hallmarks_trainer import (
        train_loop_survival_hallmarks,
        train_loop_survival_hallmarks_mb,
        validate_survival_hallmarks,
        validate_survival_hallmarks_mb
    )

    # Training loop - NO VALIDATION during training
    print(f'\nStarting training for {args.max_epochs} epochs...')
    for epoch in range(args.max_epochs):
        if args.use_micro_batch:
            train_loop_survival_hallmarks_mb(
                epoch, args.bs_micro, model, train_loader, optimizer,
                args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg,
                args.gc, args
            )
        else:
            train_loop_survival_hallmarks(
                epoch, model, train_loader, optimizer,
                args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg,
                args.gc, args
            )

    # Evaluate on test set at final epoch
    print('\n' + '=' * 40)
    print('Final Evaluation on Test Set')
    print('=' * 40)

    # Collect train events/times for IBS computation
    train_events = (1 - train_split.patients_df['dss_censorship'].values).astype(bool)
    train_times = train_split.patients_df['dss_survival_days'].values.astype(float)
    ibs_bins = train_split.bins

    if args.use_micro_batch:
        test_results, c_index_test, ibs_test = validate_survival_hallmarks_mb(
            fold, args.max_epochs - 1, args.bs_micro, model, test_loader,
            args.n_classes, loss_fn, reg_fn, args.lambda_reg,
            args.results_dir, args,
            train_events=train_events, train_times=train_times, bins=ibs_bins
        )
    else:
        test_results, c_index_test, ibs_test = validate_survival_hallmarks(
            fold, args.max_epochs - 1, model, test_loader,
            args.n_classes, loss_fn, reg_fn, args.lambda_reg,
            args.results_dir, args,
            train_events=train_events, train_times=train_times, bins=ibs_bins
        )

    # Save final model
    save_path = os.path.join(args.results_dir, f's_{fold}_final.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    if writer:
        writer.close()

    print(f"\n{'='*40}")
    print(f"Fold {fold} completed. Test C-Index: {c_index_test:.4f}, IBS: {ibs_test:.4f}")
    print(f"{'='*40}")

    return test_results, {'result': (c_index_test, ibs_test, args.max_epochs - 1)}


def main(args):
    # Create base dataset (loads TITAN features, RNA data, signatures once)
    print("\nLoading base dataset...")
    dataset = Generic_MIL_Survival_Dataset_Hallmarks(
        data_dir=args.data_dir,
        cancer_type=args.cancer_type,
        mode='coattn',
        n_bins=4,
        print_info=True
    )

    # Determine folds to run
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    folds = np.arange(start, end)
    summary_all_folds = {}

    # Train each fold
    for fold in folds:
        start_t = timer()
        seed_torch(args.seed)

        results_pkl_path = os.path.join(
            args.results_dir, f'split_{fold}_test_results.pkl'
        )

        # Skip if already done
        if os.path.isfile(results_pkl_path) and not args.overwrite:
            print(f"Skipping Fold {fold} (already completed)")
            continue

        # Train fold
        test_results, print_results = train_fold(dataset, fold, args)

        # Save results
        save_pkl(results_pkl_path, test_results)
        summary_all_folds[fold] = print_results

        end_t = timer()
        print(f'Fold {fold} Time: {end_t - start_t:.2f} seconds')

    # Print summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    result_cindex = []
    result_ibs = []
    for fold in summary_all_folds:
        c_index = summary_all_folds[fold]['result'][0]
        ibs = summary_all_folds[fold]['result'][1]
        print(f"Fold {fold}, C-Index: {c_index:.4f}, IBS: {ibs:.4f}")
        result_cindex.append(c_index)
        result_ibs.append(ibs)

    if len(result_cindex) > 0:
        result_cindex = np.array(result_cindex)
        result_ibs = np.array(result_ibs)
        print(f"\nAvg C-Index of {len(summary_all_folds)} folds: {result_cindex.mean():.3f}")
        print(f"Std (population): {result_cindex.std():.3f}")
        print(f"Std (sample): {result_cindex.std(ddof=1):.3f}")
        valid_ibs = result_ibs[~np.isnan(result_ibs)]
        if len(valid_ibs) > 0:
            print(f"Avg IBS of {len(valid_ibs)} folds: {valid_ibs.mean():.4f}")
            print(f"IBS Std (sample): {valid_ibs.std(ddof=1):.4f}" if len(valid_ibs) > 1 else "")


def run_single_cancer(cancer_type, args):
    """Run experiment for a single cancer type"""
    # Update args for this cancer type
    args.cancer_type = cancer_type

    # Build experiment code
    exp_code = f'TCGA_{args.cancer_type}_hallmarks_{args.model_type}_s{args.seed}'
    if args.model_type == 'motcat':
        if 'uot' in args.ot_impl:
            exp_code += f'_UOT_reg{args.ot_reg}_tau{args.ot_tau}'
        elif 'sinkhorn' in args.ot_impl:
            exp_code += f'_SINKHORN_reg{args.ot_reg}'

    print("\n" + "=" * 80)
    print(f"Experiment: {exp_code}")
    print(f"Cancer Type: {args.cancer_type}")
    print(f"Model: {args.model_type}")
    print(f"Max Epochs: {args.max_epochs}")
    print("=" * 80)

    # Create results directory for this cancer type
    cancer_results_dir = os.path.join(args.results_dir, exp_code)
    if not os.path.isdir(cancer_results_dir):
        os.makedirs(cancer_results_dir)
    print(f"Results will be saved to: {cancer_results_dir}")

    # Check if already completed (skip if not overwrite)
    if 'summary_latest.csv' in os.listdir(cancer_results_dir) and not args.overwrite:
        print(f"Experiment <{exp_code}> already exists! Skipping.")
        return None

    # Store original results_dir and update for this cancer
    original_results_dir = args.results_dir
    args.results_dir = cancer_results_dir

    # Run main training
    try:
        main(args)
    finally:
        # Restore original results_dir
        args.results_dir = original_results_dir

    return exp_code


if __name__ == "__main__":
    # All supported cancer types
    ALL_CANCER_TYPES = ['BLCA', 'BRCA', 'KIRC', 'LUAD', 'STAD', 'COADREAD']

    start = timer()

    if args.cancer_type == 'ALL':
        # Run all 6 cancer types
        print("=" * 80)
        print("Running ALL 6 cancer types with 5-fold cross-validation")
        print(f"Cancer types: {ALL_CANCER_TYPES}")
        print("=" * 80)

        completed = []
        failed = []

        for cancer_type in ALL_CANCER_TYPES:
            cancer_start = timer()
            try:
                exp_code = run_single_cancer(cancer_type, args)
                if exp_code:
                    completed.append(cancer_type)
            except Exception as e:
                print(f"\nERROR: Failed to run {cancer_type}: {e}")
                failed.append(cancer_type)
            cancer_end = timer()
            print(f"\n{cancer_type} completed in {cancer_end - cancer_start:.2f} seconds")

        # Print summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY - ALL CANCER TYPES")
        print("=" * 80)
        print(f"Completed: {completed}")
        if failed:
            print(f"Failed: {failed}")
    else:
        # Run single cancer type (original behavior)
        # Build experiment code
        exp_code = f'TCGA_{args.cancer_type}_hallmarks_{args.model_type}_s{args.seed}'
        if args.model_type == 'motcat':
            if 'uot' in args.ot_impl:
                exp_code += f'_UOT_reg{args.ot_reg}_tau{args.ot_tau}'
            elif 'sinkhorn' in args.ot_impl:
                exp_code += f'_SINKHORN_reg{args.ot_reg}'

        print("=" * 80)
        print(f"Experiment: {exp_code}")
        print(f"Cancer Type: {args.cancer_type}")
        print(f"Model: {args.model_type}")
        print(f"Max Epochs: {args.max_epochs}")
        print("=" * 80)

        # Create results directory
        args.results_dir = os.path.join(args.results_dir, exp_code)
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)
        print(f"Results will be saved to: {args.results_dir}")

        # Check if already completed
        if 'summary_latest.csv' in os.listdir(args.results_dir) and not args.overwrite:
            print(f"Experiment <{exp_code}> already exists! Exiting.")
            sys.exit()

        main(args)

    end = timer()
    print(f"\nTotal Time: {end - start:.2f} seconds")
    print("Done!")
