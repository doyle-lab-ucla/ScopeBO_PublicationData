#################################################################################################
# This code was adapted from (https://github.com/DeepSynthesis/ScopeMap/tree/main/SubstrateScore)
# slight modifications were made for use in a Jupyter notebook, and they are marked with comments
#################################################################################################


import numpy as np
import pandas as pd
import argparse
import os
from .score_utils import calc_entropy, calc_mean_squared_distance
from .sampling import cvt_sampling_df_norepeat, kennard_stone_sampling_df_norepeat, rand_sampling_df_no_repeat
from .desc_gen_util import spoc_descriptors


def generate_fingerprint_file_if_needed(data_file, fp_file):
    """Generate fingerprint file if not provided or doesn't exist."""
    if fp_file is None:
        base_name = os.path.splitext(os.path.basename(data_file))[0]
        fp_file = f"fp_spoc_morgan41024_Maccs_{base_name}.csv"
    
    if not os.path.exists(fp_file):
        print(f"   - Fingerprint file '{fp_file}' not found, generating...")
        spoc_descriptors(data_file)
        print(f"   - Generated fingerprint file: {fp_file}")
    else:
        print(f"   - Using existing fingerprint file: {fp_file}")
    
    return fp_file


def parse_args(argv=None):  # added argv=None
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate sampling quality using different sampling methods.')
    
    parser.add_argument('--substrate-file', type=str,
                        help='Path to substrate data file')
    parser.add_argument('--substrate-fp-file', type=str, default=None,
                        help='Path to substrate fingerprint file (optional, will be generated if not provided)')
    parser.add_argument('--experimental-file', type=str,
                        help='Path to experimental data file')
    parser.add_argument('--experimental-fp-file', type=str, default=None,
                        help='Path to experimental fingerprint file (optional, will be generated if not provided)')
    
    parser.add_argument('--distance-metric', type=str, default='euclidean',
                        choices=['euclidean', 'manhattan', 'cosine', 'tanimoto'],
                        help='Distance metric for calculations (default: euclidean)')
    parser.add_argument('--k-neighbors', type=int, default=5,
                        help='Number of nearest neighbors for entropy calculation (default: 5)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--random-runs', type=int, default=5,
                        help='Number of random sampling runs (default: 5)')
    
    parser.add_argument('--not-feature-columns', nargs='+', default=['smiles'],
                        help='List of non-feature columns to exclude (default: [smiles])')
    
    return parser.parse_args(argv)  # added argv to parse_args()


def main(argv=None):  # added argv=None
    args = parse_args(argv)  # added argv to parse_args()
    print(f"Arguments: {args}")
    
    np.random.seed(args.random_seed)
    
    print("=" * 80)
    print("Starting Sampling Quality Evaluation")
    print("=" * 80)
    
    print("\n【Part 1: CVT and Kennard-Stone Sampling Evaluation】")
    print("-" * 80)
    
    print(f"\n1. Reading data files...")
    
    experimental_df = pd.read_csv(args.experimental_file)
    sampling_size = len(experimental_df)
    print(f"   - Experimental data: {experimental_df.shape}")
    print(f"   - Sampling size set to experimental data length: {sampling_size}")
    
    substrate_df = pd.read_csv(args.substrate_file)
    
    substrate_fp_file = generate_fingerprint_file_if_needed(args.substrate_file, args.substrate_fp_file)
    substrate_fp_df = pd.read_csv(substrate_fp_file)
    
    complete_space_df = pd.concat([substrate_df, substrate_fp_df], axis=1)
    print(f"   - Combined complete space: {complete_space_df.shape}")
    
    print(f"\n2. Performing CVT sampling (sample size: {sampling_size})...")
    cvt_sampling_df, _ = cvt_sampling_df_norepeat(
        data=complete_space_df,
        k=sampling_size,
        not_feature_columns=args.not_feature_columns,
        distance_metric=args.distance_metric
    )
    print(f"   - CVT sampling completed, sample count: {len(cvt_sampling_df)}")
    
    print(f"\n3. Calculating CVT sampling evaluation metrics...")
    cvt_entropy = calc_entropy(
        complete_space_df=complete_space_df,
        sampling_points_df=cvt_sampling_df,
        not_feature_col=args.not_feature_columns,
        k=args.k_neighbors,
        distance_metric=args.distance_metric
    )
    cvt_msd = calc_mean_squared_distance(
        complete_space_df=complete_space_df,
        sampling_points_df=cvt_sampling_df,
        not_feature_col=args.not_feature_columns,
        distance_metric=args.distance_metric
    )
    print(f"   - CVT sampling Entropy: {cvt_entropy:.6f}")
    print(f"   - CVT sampling MSD: {cvt_msd:.6f}")
    
    print(f"\n4. Performing Kennard-Stone sampling (sample size: {sampling_size})...")
    ks_sampling_df, _ = kennard_stone_sampling_df_norepeat(
        data=complete_space_df,
        k=sampling_size,
        not_feature_columns=args.not_feature_columns,
        distance_metric=args.distance_metric
    )
    print(f"   - Kennard-Stone sampling completed, sample count: {len(ks_sampling_df)}")
    
    print(f"\n5. Calculating Kennard-Stone sampling evaluation metrics...")
    ks_entropy = calc_entropy(
        complete_space_df=complete_space_df,
        sampling_points_df=ks_sampling_df,
        not_feature_col=args.not_feature_columns,
        k=args.k_neighbors,
        distance_metric=args.distance_metric
    )
    ks_msd = calc_mean_squared_distance(
        complete_space_df=complete_space_df,
        sampling_points_df=ks_sampling_df,
        not_feature_col=args.not_feature_columns,
        distance_metric=args.distance_metric
    )
    print(f"   - Kennard-Stone sampling Entropy: {ks_entropy:.6f}")
    print(f"   - Kennard-Stone sampling MSD: {ks_msd:.6f}")
    
    print(f"\n6. Performing random sampling (sample size: {sampling_size}, {args.random_runs} runs)...")
    rand_entropies = []
    rand_msds = []
    
    for i in range(args.random_runs):
        np.random.seed(args.random_seed + i)
        rand_sampling_df, _ = rand_sampling_df_no_repeat(
            data=complete_space_df,
            k=sampling_size,
            not_feature_columns=args.not_feature_columns,
            distance_metric=args.distance_metric
        )
        
        rand_entropy = calc_entropy(
            complete_space_df=complete_space_df,
            sampling_points_df=rand_sampling_df,
            not_feature_col=args.not_feature_columns,
            k=args.k_neighbors,
            distance_metric=args.distance_metric
        )
        rand_msd = calc_mean_squared_distance(
            complete_space_df=complete_space_df,
            sampling_points_df=rand_sampling_df,
            not_feature_col=args.not_feature_columns,
            distance_metric=args.distance_metric
        )
        
        rand_entropies.append(rand_entropy)
        rand_msds.append(rand_msd)
        print(f"   - Run {i+1}: Entropy={rand_entropy:.6f}, MSD={rand_msd:.6f}")
    
    rand_entropy_mean = np.mean(rand_entropies)
    rand_entropy_std = np.std(rand_entropies, ddof=1)
    rand_msd_mean = np.mean(rand_msds)
    rand_msd_std = np.std(rand_msds, ddof=1)
    
    print(f"\n7. Random sampling statistics:")
    print(f"   - Mean Entropy: {rand_entropy_mean:.6f} ± {rand_entropy_std:.6f}")
    print(f"   - Mean MSD: {rand_msd_mean:.6f} ± {rand_msd_std:.6f}")
    
    print("\n\n【Part 2: Experimental Data Evaluation】")
    print("-" * 80)
    
    print(f"\n8. Processing experimental data fingerprints...")
    
    experimental_fp_file = generate_fingerprint_file_if_needed(args.experimental_file, args.experimental_fp_file)
    experimental_fp_df = pd.read_csv(experimental_fp_file)
    
    print(f"   - Experimental data features: {experimental_fp_df.shape}")
    
    if 'SMILES' in experimental_df.columns:
        experimental_df = experimental_df.rename(columns={'SMILES': 'smiles'})
    experimental_combined_df = pd.concat([experimental_df[args.not_feature_columns], experimental_fp_df], axis=1)
    print(f"   - Combined experimental data: {experimental_combined_df.shape}")
    
    print(f"\n9. Calculating experimental sampling evaluation metrics...")
    exp_entropy = calc_entropy(
        complete_space_df=complete_space_df,
        sampling_points_df=experimental_combined_df,
        not_feature_col=args.not_feature_columns,
        k=args.k_neighbors,
        distance_metric=args.distance_metric
    )
    exp_msd = calc_mean_squared_distance(
        complete_space_df=complete_space_df,
        sampling_points_df=experimental_combined_df,
        not_feature_col=args.not_feature_columns,
        distance_metric=args.distance_metric
    )
    print(f"   - Experimental sampling Entropy: {exp_entropy:.6f}")
    print(f"   - Experimental sampling MSD: {exp_msd:.6f}")
    
    print("\n\n【Part 3: U-Score and R-Score Calculation】")
    print("-" * 80)
    
    print(f"\n10. Calculating U-Score...")
    
    if ks_entropy != rand_entropy_mean:
        u_score = 60 + 40 * (exp_entropy - rand_entropy_mean) / (ks_entropy - rand_entropy_mean)
        print(f"   - U-Score: {u_score:.6f}")
    else:
        u_score = None
        print(f"   - U-Score: Unable to calculate (K-S entropy equals random entropy)")
    
    print(f"\n11. Calculating R-Score...")
    
    if cvt_msd != rand_msd_mean:
        r_score = 60 + 40 * (exp_msd - rand_msd_mean) / (cvt_msd - rand_msd_mean)
        print(f"   - R-Score: {r_score:.6f}")
    else:
        r_score = None
        print(f"   - R-Score: Unable to calculate (CVT MSD equals random MSD)")
    
    print("\n\n" + "=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    print(f"\n{'Method':<25} {'Entropy':<30} {'MSD':<20}")
    print("-" * 80)
    print(f"{'CVT Sampling':<25} {cvt_entropy:<30.6f} {cvt_msd:<20.6f}")
    print(f"{'Kennard-Stone Sampling':<25} {ks_entropy:<30.6f} {ks_msd:<20.6f}")
    print(f"{'Random Sampling (Mean)':<25} {rand_entropy_mean:<30.6f} {rand_msd_mean:<20.6f}")
    print(f"{'Random Sampling (Std)':<25} {rand_entropy_std:<30.6f} {rand_msd_std:<20.6f}")
    print(f"{'Experimental Sampling':<25} {exp_entropy:<30.6f} {exp_msd:<20.6f}")
    
    print("\n" + "-" * 80)
    print("Evaluation Metrics")
    print("-" * 80)
    if u_score is not None:
        print(f"{'U-Score':<25} {u_score:<30.6f}")
    else:
        print(f"{'U-Score':<25} {'Unable to calculate':<30}")
    
    if r_score is not None:
        print(f"{'R-Score':<25} {r_score:<30.6f}")
    else:
        print(f"{'R-Score':<25} {'Unable to calculate':<30}")
    
    print("=" * 80)

    return u_score, r_score  # added return statement for U-Score and R-Score


if __name__ == "__main__":
    main()
