# analysis.py
""" Analysis, evaluation, and plotting functions """

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (precision_score, recall_score, f1_score,
                                roc_curve, roc_auc_score,
                                average_precision_score, precision_recall_curve)
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from tqdm import tqdm


def calculate_static_distributions_ground_truth(answers_data, question_categories_data, config):
    """ Calculates static distributions from answers (Ground Truth for Eval). """
    print("Calculating static RIASEC distributions (Ground Truth for Eval)...")
    num_samples = config.NUM_SAMPLES
    num_categories = config.NUM_CATEGORIES
    static_distributions_np = np.zeros((num_samples, num_categories))
    dominant_static_category = np.zeros(num_samples, dtype=int)

    # Ensure tensors are on CPU for numpy conversion if needed
    answers_data_cpu = answers_data.cpu()
    question_categories_data_cpu = question_categories_data.cpu()

    for i in range(num_samples):
        answers_i = answers_data_cpu[i]
        yes_indices = torch.where(answers_i == 1)[0]
        num_yes = len(yes_indices)
        if num_yes > 0:
            # Ensure indices are within bounds
            valid_indices = yes_indices[yes_indices < len(question_categories_data_cpu)]
            if len(valid_indices) > 0:
                categories_for_yes = question_categories_data_cpu[valid_indices]
                counts = torch.bincount(categories_for_yes, minlength=num_categories)
                dist = counts.float() / num_yes
                static_distributions_np[i] = dist.numpy()
                if len(dist) > 0: # Check if dist is not empty
                   dominant_static_category[i] = torch.argmax(dist).item()
                else:
                   dominant_static_category[i] = -1 # Or some placeholder
            else:
                dominant_static_category[i] = -1 # Or some placeholder if no valid indices
        else:
            dominant_static_category[i] = -1 # Assign placeholder if no 'yes' answers

    print("Ground truth static distributions and dominant categories calculated.")
    return static_distributions_np, dominant_static_category


def get_model_outputs_analysis(model, dataloader, categories_tensor, device):
    """ Runs model inference and collects outputs for analysis. """
    model.eval()
    model.to(device)
    all_mu, all_logvar, all_recon_probs, all_original_answers, all_predicted_static_logits = [], [], [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating Model", leave=False)
        for (answers_batch,) in pbar: # Dataloader only provides answers for eval
            answers_batch = answers_batch.to(device)
            recon_logits, mu, logvar, pred_static_logits = model(answers_batch, categories_tensor)

            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_recon_probs.append(torch.sigmoid(recon_logits).cpu())
            all_original_answers.append(answers_batch.cpu())
            all_predicted_static_logits.append(pred_static_logits.cpu())

    results = {
        "original": torch.cat(all_original_answers),
        "mu": torch.cat(all_mu),
        "logvar": torch.cat(all_logvar),
        "recon_probs": torch.cat(all_recon_probs),
        "predicted_static_logits": torch.cat(all_predicted_static_logits)
    }
    return results

def compare_vectors(static_distributions_np, latent_np, config):
    """ Prints side-by-side comparison of static dist and latent vector. """
    print(f"\n--- Side-by-Side Vector Comparison (First {config.NUM_VECTORS_COMPARE} Samples) ---")
    num_display = min(config.NUM_VECTORS_COMPARE, len(static_distributions_np))
    latent_space_name = "VAE mu" # Specific to this model type

    for i in range(num_display):
        static_vector = static_distributions_np[i]
        latent_vector = latent_np[i]
        print(f"\n--- Sample {i} ---")
        print(f"Static Dist: [{np.array2string(static_vector, formatter={'float_kind':lambda x: '%.4f' % x}, separator=' ')}]")
        print(f"{latent_space_name}: [{np.array2string(latent_vector, formatter={'float_kind':lambda x: '%.4f' % x}, separator=' ')}]")
    print("\n--- End Vector Comparison ---")


def run_analysis(model, full_loader, answers_data, question_categories_data, categories_tensor, device, config):
    """ Performs the complete analysis pipeline. """
    print("\n--- Starting Analysis ---")

    # 1. Get Model Outputs
    model_outputs = get_model_outputs_analysis(model, full_loader, categories_tensor, device)
    mu_np = model_outputs["mu"].numpy()
    recon_probs_np = model_outputs["recon_probs"].numpy()
    original_answers_np = model_outputs["original"].numpy()
    predicted_static_logits_np = model_outputs["predicted_static_logits"].numpy()
    # Optionally convert predicted logits to probabilities if needed for comparison
    # predicted_static_probs_np = torch.softmax(model_outputs["predicted_static_logits"], dim=1).numpy()

    # 2. Calculate Ground Truth Static Distributions for Eval
    static_distributions_np_gt, dominant_static_category_gt = calculate_static_distributions_ground_truth(
        model_outputs["original"], # Use original answers from this run
        question_categories_data,
        config
    )

    # 3. Calculate Metrics
    print("\n--- Analytical Statistics ---")
    # Reconstruction Accuracy
    recon_binary_np = (recon_probs_np > 0.5).astype(int)
    accuracy = accuracy_score(original_answers_np.flatten(), recon_binary_np.flatten())
    print(f"Overall Reconstruction Accuracy: {accuracy:.4f}")

    # Static Distribution Prediction MSE (Logits vs Ground Truth Distribution)
    static_pred_mse = mean_squared_error(predicted_static_logits_np, static_distributions_np_gt)
    print(f"Static Prediction MSE (logits vs target dist): {static_pred_mse:.4f}")

    # Clustering Alignment (K-Means on mu vs. Dominant Static Category)
    print(f"\nComparing K-Means ({config.N_CLUSTERS_EVAL} clusters on mu):")
    # Filter out samples with no dominant category if any exist (-1 was placeholder)
    valid_indices = np.where(dominant_static_category_gt != -1)[0]
    if len(valid_indices) < config.N_CLUSTERS_EVAL:
         print(f"Warning: Only {len(valid_indices)} samples with valid dominant categories found. Skipping K-Means/ARI.")
         ari_score = np.nan
         kmeans_centers = None # No centers if K-Means not run
    else:
        mu_np_valid = mu_np[valid_indices]
        dominant_static_category_gt_valid = dominant_static_category_gt[valid_indices]

        # Ensure n_clusters is not more than the number of valid samples
        n_clusters_actual = min(config.N_CLUSTERS_EVAL, len(mu_np_valid))
        if n_clusters_actual < config.N_CLUSTERS_EVAL:
            print(f"Warning: Reducing K-Means clusters to {n_clusters_actual} due to insufficient valid samples.")

        if n_clusters_actual >= 2: # K-Means needs at least 2 clusters
            kmeans_mu = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
            clusters_mu = kmeans_mu.fit_predict(mu_np_valid)
            ari_score = adjusted_rand_score(dominant_static_category_gt_valid, clusters_mu)
            print(f"Adjusted Rand Index (ARI) between GT static category and mu clusters: {ari_score:.4f}")
            kmeans_centers = kmeans_mu.cluster_centers_ # Store centers
        else:
            print("Skipping K-Means/ARI: Not enough clusters possible.")
            ari_score = np.nan
            kmeans_centers = None

    # Then modify the run_analysis function to add the new metrics section
    # Add after line 125 (after calculating current metrics)

    # Binary Classification Metrics for Reconstruction
    print("\n--- Detailed Reconstruction Metrics ---")
    precision = precision_score(original_answers_np.flatten(), recon_binary_np.flatten())
    recall = recall_score(original_answers_np.flatten(), recon_binary_np.flatten())
    f1 = f1_score(original_answers_np.flatten(), recon_binary_np.flatten())
    print(f"Reconstruction Precision: {precision:.4f}")
    print(f"Reconstruction Recall (Sensitivity): {recall:.4f}")
    print(f"Reconstruction F1 Score: {f1:.4f}")

    # ROC and Precision-Recall Metrics
    auroc = roc_auc_score(original_answers_np.flatten(), recon_probs_np.flatten())
    print(f"Reconstruction AUROC: {auroc:.4f}")

    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(original_answers_np.flatten(), recon_probs_np.flatten())

    # Find sensitivity at specific specificity levels
    for target_spec in [0.9, 0.95, 0.99]:
        specificity = 1 - fpr
        idx = np.argmin(np.abs(specificity - target_spec))
        sensitivity_at_specificity = tpr[idx]
        threshold_at_target = thresholds[idx]
        print(f"Sensitivity at {target_spec:.2f} Specificity: {sensitivity_at_specificity:.4f} (threshold: {threshold_at_target:.4f})")

    # Average Precision
    average_precision = average_precision_score(original_answers_np.flatten(), recon_probs_np.flatten())
    print(f"Average Precision: {average_precision:.4f}")

    # Plot Precision-Recall curve
    precision_values, recall_values, _ = precision_recall_curve(original_answers_np.flatten(), recon_probs_np.flatten())
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (AP={:.4f})'.format(average_precision))
    plt.grid(True)
    plt.savefig(config.PLOT_FILENAME_PR_CURVE)
    print(f"Saved Precision-Recall curve to {config.PLOT_FILENAME_PR_CURVE}")
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (AUC={:.4f})'.format(auroc))
    plt.grid(True)
    plt.savefig(config.PLOT_FILENAME_ROC_CURVE)
    print(f"Saved ROC curve to {config.PLOT_FILENAME_ROC_CURVE}")
    plt.close()

    # Static Distribution Evaluation
    print("\n--- Detailed Static Distribution Metrics ---")
    # Convert logits to probabilities for better comparison
    predicted_static_probs_np = torch.softmax(model_outputs["predicted_static_logits"], dim=1).numpy()

    # Category-wise correlation
    category_correlations = []
    for cat in range(config.NUM_CATEGORIES):
        pearson_corr, _ = pearsonr(static_distributions_np_gt[:, cat], predicted_static_probs_np[:, cat])
        category_correlations.append(pearson_corr)
        print(f"Category {cat} Pearson correlation: {pearson_corr:.4f}")

    avg_category_correlation = np.mean(category_correlations)
    print(f"Average category-wise Pearson correlation: {avg_category_correlation:.4f}")

    # R-squared for overall distribution fit
    r2 = r2_score(static_distributions_np_gt.flatten(), predicted_static_probs_np.flatten())
    print(f"Static Distribution R-squared: {r2:.4f}")


    # 4. Side-by-Side Vector Comparison
    compare_vectors(static_distributions_np_gt, mu_np, config)

    # 5. Dimensionality Reduction and Visualization
    print("\n--- Generating Visualizations ---")
    plt.style.use('seaborn-v0_8-darkgrid')

    # Reduce dimensions (only on valid samples if filtering occurred)
    if len(valid_indices) == len(mu_np): # No filtering needed
        mu_to_reduce = mu_np
        static_to_reduce = static_distributions_np_gt
        colors_gt = dominant_static_category_gt
    else:
        mu_to_reduce = mu_np_valid
        static_to_reduce = static_distributions_np_gt[valid_indices]
        colors_gt = dominant_static_category_gt_valid


    if len(mu_to_reduce) > 2: # Need samples for PCA
        # --- 2D Plots ---
        pca_2d = PCA(n_components=2, random_state=42)
        mu_reduced_2d = pca_2d.fit_transform(mu_to_reduce)
        static_reduced_2d = pca_2d.fit_transform(static_to_reduce) # Use same PCA object or fit separately? Fit separately usually better.
        pca_2d_static = PCA(n_components=2, random_state=42)
        static_reduced_2d = pca_2d_static.fit_transform(static_to_reduce)


        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        # Plot 1: Latent Space (mu) colored by GT Static Category
        scatter1 = axes[0].scatter(mu_reduced_2d[:, 0], mu_reduced_2d[:, 1], c=colors_gt, cmap='viridis', alpha=0.7, s=15)
        axes[0].set_title(f'VAE Latent Space (mu) ({config.LATENT_DIM}D -> 2D PCA)\nColored by Dominant Static Category (GT)')
        axes[0].set_xlabel("PCA Component 1"); axes[0].set_ylabel("PCA Component 2")
        legend1 = axes[0].legend(*scatter1.legend_elements(num=config.NUM_CATEGORIES), title="Static Cat.")
        axes[0].add_artist(legend1)
        # Plot 2: Static Distributions colored by GT Static Category
        scatter2 = axes[1].scatter(static_reduced_2d[:, 0], static_reduced_2d[:, 1], c=colors_gt, cmap='viridis', alpha=0.7, s=15)
        axes[1].set_title(f'Static Distributions (GT) ({config.NUM_CATEGORIES}D -> 2D PCA)\nColored by Dominant Static Category (GT)')
        axes[1].set_xlabel("PCA Component 1"); axes[1].set_ylabel("PCA Component 2")
        legend2 = axes[1].legend(*scatter2.legend_elements(num=config.NUM_CATEGORIES), title="Static Cat.")
        axes[1].add_artist(legend2)
        plt.suptitle('Transformer VAE + Static Head Analysis', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(config.PLOT_FILENAME_2D)
        print(f"Saved 2D analysis plots to {config.PLOT_FILENAME_2D}")
        plt.close(fig) # Close the figure

        # --- 3D Plots ---
        if len(mu_to_reduce) > 3: # Need samples for 3D PCA
            pca_3d = PCA(n_components=3, random_state=42)
            mu_reduced_3d = pca_3d.fit_transform(mu_to_reduce)
            pca_3d_static = PCA(n_components=3, random_state=42)
            static_reduced_3d = pca_3d_static.fit_transform(static_to_reduce)
            hover_text_static = [f"Sample: {i}<br>Static Cat: {cat}" for i, cat in zip(valid_indices, colors_gt)] # Use original index if filtered
            category_labels_map_static = {i: f"Static Cat {i}" for i in range(config.NUM_CATEGORIES)}

            # Plot 3D Latent Space (mu) colored by GT Static Cat
            fig_3d_latent = go.Figure()
            for cat_idx in range(config.NUM_CATEGORIES):
                indices_3d = np.where(colors_gt == cat_idx)[0]
                if len(indices_3d) > 0:
                    fig_3d_latent.add_trace(go.Scatter3d(x=mu_reduced_3d[indices_3d, 0], y=mu_reduced_3d[indices_3d, 1], z=mu_reduced_3d[indices_3d, 2], mode='markers', marker=dict(size=3, opacity=0.7), text=[hover_text_static[i] for i in indices_3d], hoverinfo='text', name=category_labels_map_static[cat_idx]))
            fig_3d_latent.update_layout(title=f'VAE Latent Space (mu) ({config.LATENT_DIM}D -> 3D PCA) Colored by Static Category (GT)', scene=dict(xaxis_title='PCA Comp 1', yaxis_title='PCA Comp 2', zaxis_title='PCA Comp 3'), margin=dict(l=0, r=0, b=0, t=40))
            fig_3d_latent.write_html(config.PLOT_FILENAME_3D_LATENT)
            print(f"Saved interactive 3D latent space plot to {config.PLOT_FILENAME_3D_LATENT}")

            # Plot 3D Static Distributions colored by GT Static Cat
            fig_3d_static = go.Figure()
            for cat_idx in range(config.NUM_CATEGORIES):
                 indices_3d_static = np.where(colors_gt == cat_idx)[0]
                 if len(indices_3d_static) > 0:
                    fig_3d_static.add_trace(go.Scatter3d(x=static_reduced_3d[indices_3d_static, 0], y=static_reduced_3d[indices_3d_static, 1], z=static_reduced_3d[indices_3d_static, 2], mode='markers', marker=dict(size=3, opacity=0.7), text=[hover_text_static[i] for i in indices_3d_static], hoverinfo='text', name=category_labels_map_static[cat_idx]))
            fig_3d_static.update_layout(title=f'Static Distributions (GT) ({config.NUM_CATEGORIES}D -> 3D PCA) Colored by Static Category (GT)', scene=dict(xaxis_title='PCA Comp 1', yaxis_title='PCA Comp 2', zaxis_title='PCA Comp 3'), margin=dict(l=0, r=0, b=0, t=40))
            fig_3d_static.write_html(config.PLOT_FILENAME_3D_STATIC)
            print(f"Saved interactive 3D static distributions plot to {config.PLOT_FILENAME_3D_STATIC}")

        # --- Correlation Heatmaps ---
        # Latent Space (mu) Dimensions
        mu_df = pd.DataFrame(mu_np, columns=[f'Latent_{i}' for i in range(config.LATENT_DIM)])
        corr_matrix_mu = mu_df.corr()
        plt.figure(figsize=(8, 6)); sns.heatmap(corr_matrix_mu, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Correlation Matrix of VAE Latent Space (mu) Dimensions'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        plt.savefig(config.PLOT_FILENAME_CORR_LATENT)
        print(f"Saved latent dimension correlation heatmap to {config.PLOT_FILENAME_CORR_LATENT}")
        plt.close()

        # Cosine Similarity Between K-Means Centroids (if calculated)
        if kmeans_centers is not None:
            cosine_sim_matrix = cosine_similarity(kmeans_centers)
            centroid_labels = [f"KMeans Centroid {i}" for i in range(n_clusters_actual)]
            plt.figure(figsize=(8, 6)); sns.heatmap(cosine_sim_matrix, annot=True, cmap='viridis', fmt='.2f', xticklabels=centroid_labels, yticklabels=centroid_labels, linewidths=.5)
            plt.title('Cosine Similarity Between K-Means Centroids (on mu)'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
            plt.savefig(config.PLOT_FILENAME_COSINE_CENTROIDS)
            print(f"Saved K-Means centroid cosine similarity heatmap to {config.PLOT_FILENAME_COSINE_CENTROIDS}")
            plt.close()
        else:
            print("Skipping K-Means centroid cosine similarity plot.")

    else:
        print("Skipping PCA/Plotting due to insufficient valid samples.")


    plt.close('all') # Close any remaining figures
    print("--- Analysis Finished ---")
