# participant_analysis.py
""" Script for analyzing specific participants from dataset.csv """

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse

# Import from existing modules
import config
import train
from model import TransformerVAEStaticHead
from data import calculate_target_static_distributions, prepare_dataloaders
from analysis import calculate_static_distributions_ground_truth

def load_dataset(csv_file_path):
    """
    Load the dataset from a CSV file

    Parameters:
    csv_file_path (str): Path to the CSV file

    Returns:
    tuple: (df, participant_info, answers_data)
    """
    print(f"Loading data from {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    # Extract participant information
    participant_info = df[['name', 'student_id', 'timestamp']].copy()

    # Extract question responses (q1 to q30)
    question_columns = [col for col in df.columns if col.startswith('q')]
    answer_df = df[question_columns]

    # Convert to torch tensor
    answers_data = torch.tensor(answer_df.values, dtype=torch.int)

    return df, participant_info, answers_data

def load_or_train_model(answers_data, question_categories_data, device, model_path=None):
    """
    Load a pre-trained model or train a new one if no model file exists

    Parameters:
    answers_data (torch.Tensor): Answer data tensor
    question_categories_data (torch.Tensor): Question category assignments
    device (torch.device): Device to run model on
    model_path (str): Path to save/load model

    Returns:
    TransformerVAEStaticHead: Trained model
    """
    num_samples = answers_data.shape[0]
    num_questions = answers_data.shape[1]

    # Initialize model
    personality_model = TransformerVAEStaticHead(
        num_questions=num_questions,
        num_categories=config.NUM_CATEGORIES,
        embed_dim=config.EMBED_DIM,
        latent_dim=config.LATENT_DIM,
        hidden_dim_dec=config.HIDDEN_DIM_DEC,
        hidden_dim_static_head=config.HIDDEN_DIM_STATIC_HEAD,
        transformer_nhead=config.TRANSFORMER_NHEAD,
        transformer_nlayers=config.TRANSFORMER_NLAYERS,
        transformer_dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD
    ).to(device)

    # Check if model file exists
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        personality_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No model file found. Training a new model...")
        import torch.optim as optim
        from train import train_model

        # Calculate target static distributions for training
        target_static_distributions = calculate_target_static_distributions(
            answers_data, question_categories_data, num_samples, config.NUM_CATEGORIES
        )

        # Create dataloaders
        train_loader, _ = prepare_dataloaders(
            answers_data, target_static_distributions, config.BATCH_SIZE
        )

        # Train model
        optimizer = optim.Adam(personality_model.parameters(), lr=config.LEARNING_RATE)
        train.train_model(personality_model, train_loader, question_categories_data, optimizer, config, device)

        # Save trained model
        if model_path:
            print(f"Saving trained model to {model_path}")
            torch.save(personality_model.state_dict(), model_path)

    return personality_model

def analyze_participant(model, participant_index, answers_data, question_categories_data, participant_info, device):
    """
    Analyze a specific participant

    Parameters:
    model (TransformerVAEStaticHead): Trained model
    participant_index (int): Index of participant to analyze
    answers_data (torch.Tensor): Answer data tensor
    question_categories_data (torch.Tensor): Question category assignments
    participant_info (pd.DataFrame): Participant information
    device (torch.device): Device to run model on

    Returns:
    dict: Analysis results
    """
    model.eval()

    # Get participant answers
    participant_answers = answers_data[participant_index].unsqueeze(0).to(device)
    participant_name = participant_info.iloc[participant_index]['name']
    participant_id = participant_info.iloc[participant_index]['student_id']

    # Run model inference
    with torch.no_grad():
        recon_logits, mu, logvar, predicted_static_logits = model(participant_answers, question_categories_data)

    # Calculate static distributions (ground truth)
    static_dist_gt, dominant_category_gt = calculate_static_distributions_ground_truth(
        participant_answers.cpu(),
        question_categories_data,
        type('config', (), {'NUM_SAMPLES': 1, 'NUM_CATEGORIES': config.NUM_CATEGORIES})
    )

    # Convert to probabilities
    recon_probs = torch.sigmoid(recon_logits)
    predicted_static_probs = torch.softmax(predicted_static_logits, dim=1)

    # Get dominant category from model prediction
    dominant_category_pred = torch.argmax(predicted_static_probs, dim=1).item()

    # Results
    results = {
        'participant_name': participant_name,
        'participant_id': participant_id,
        'original_answers': participant_answers.cpu().numpy(),
        'reconstructed_probs': recon_probs.cpu().numpy(),
        'latent_mu': mu.cpu().numpy(),
        'static_distribution_gt': static_dist_gt,
        'static_distribution_pred': predicted_static_probs.cpu().numpy(),
        'dominant_category_gt': dominant_category_gt.item(),
        'dominant_category_pred': dominant_category_pred
    }

    return results

def analyze_participants_by_name(model, names, answers_data, question_categories_data, participant_info, device):
    """
    Analyze specific participants by name

    Parameters:
    model (TransformerVAEStaticHead): Trained model
    names (list): List of participant names to analyze
    answers_data (torch.Tensor): Answer data tensor
    question_categories_data (torch.Tensor): Question category assignments
    participant_info (pd.DataFrame): Participant information
    device (torch.device): Device to run model on

    Returns:
    dict: Analysis results for each participant
    """
    results = {}

    for name in names:
        # Find participant index by name (case-insensitive partial match)
        found = False
        for idx, participant_name in enumerate(participant_info['name']):
            if name.lower() in participant_name.lower():
                print(f"Found participant: {participant_name}")
                participant_results = analyze_participant(
                    model, idx, answers_data, question_categories_data, participant_info, device
                )
                results[participant_name] = participant_results
                found = True

        if not found:
            print(f"No participant found with name containing '{name}'")

    return results

def visualize_participant_results(results, category_names=None):
    """
    Visualize analysis results for a participant

    Parameters:
    results (dict): Analysis results
    category_names (list): Names of categories
    """
    if category_names is None:
        category_names = [f"Category {i}" for i in range(config.NUM_CATEGORIES)]

    for participant_name, participant_results in results.items():
        print(f"\n--- Results for {participant_name} (ID: {participant_results['participant_id']}) ---")

        # 1. Print dominant categories
        print(f"Ground Truth Dominant Category: {category_names[participant_results['dominant_category_gt']]}")
        print(f"Predicted Dominant Category: {category_names[participant_results['dominant_category_pred']]}")

        # 2. Set up plot
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Analysis for {participant_name}", fontsize=16)

        # 3. Plot static distributions
        axs[0, 0].bar(category_names, participant_results['static_distribution_gt'][0], alpha=0.6, label='Ground Truth')
        axs[0, 0].bar(category_names, participant_results['static_distribution_pred'][0], alpha=0.6, label='Predicted')
        axs[0, 0].set_title('Static Distribution Comparison')
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].legend()
        axs[0, 0].set_xticklabels(category_names, rotation=45, ha='right')

        # 4. Plot latent space
        latent_dims = participant_results['latent_mu'][0]
        axs[0, 1].bar(range(len(latent_dims)), latent_dims, alpha=0.7)
        axs[0, 1].set_title('Latent Space Representation (mu)')
        axs[0, 1].set_xlabel('Latent Dimension')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].set_xticks(range(len(latent_dims)))

        # 5. Plot answers vs. reconstructions
        original = participant_results['original_answers'][0]
        recon = participant_results['reconstructed_probs'][0]
        question_categories_data = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])

        # Sort by category
        indices = np.argsort(question_categories_data.numpy())
        sorted_original = original[indices]
        sorted_recon = recon[indices]
        sorted_categories = question_categories_data[indices].numpy()

        # Get category boundaries for vertical lines
        boundaries = []
        for i in range(1, len(sorted_categories)):
            if sorted_categories[i] != sorted_categories[i-1]:
                boundaries.append(i)

        axs[1, 0].bar(range(len(sorted_original)), sorted_original, alpha=0.6, label='Original')
        axs[1, 0].bar(range(len(sorted_recon)), sorted_recon, alpha=0.6, label='Reconstructed')
        for b in boundaries:
            axs[1, 0].axvline(x=b-0.5, color='red', linestyle='--', alpha=0.5)
        axs[1, 0].set_title('Answer Reconstruction (Sorted by Category)')
        axs[1, 0].set_xlabel('Question Index (Sorted)')
        axs[1, 0].set_ylabel('Response (0/1 or Probability)')
        axs[1, 0].legend()

        # 6. Plot a heatmap of answer correctness
        differences = np.abs(sorted_original - sorted_recon)
        im = axs[1, 1].imshow(differences.reshape(1, -1), cmap='coolwarm', aspect='auto')
        axs[1, 1].set_title('Reconstruction Error')
        axs[1, 1].set_xlabel('Question Index (Sorted by Category)')
        axs[1, 1].set_yticks([])
        for b in boundaries:
            axs[1, 1].axvline(x=b-0.5, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(im, ax=axs[1, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"participants/{participant_name.replace(' ', '_')}_analysis.png")
        print(f"Visualization saved to {participant_name.replace(' ', '_')}_analysis.png")

        # 7. Detailed text report
        print("\nDetailed Analysis:")
        print(f"  - Overall reconstruction accuracy: {1 - np.mean(np.abs(original - (recon > 0.5).astype(int))):.4f}")

        # Calculate per-category metrics
        print("\nCategory-specific metrics:")
        for cat_idx in range(config.NUM_CATEGORIES):
            cat_questions = np.where(question_categories_data.numpy() == cat_idx)[0]
            if len(cat_questions) > 0:
                cat_original = original[cat_questions]
                cat_recon = recon[cat_questions]
                cat_acc = 1 - np.mean(np.abs(cat_original - (cat_recon > 0.5).astype(int)))
                print(f"  - {category_names[cat_idx]}: {len(cat_questions)} questions, accuracy: {cat_acc:.4f}")

        print("\nLatent space values:")
        for i, val in enumerate(latent_dims):
            print(f"  - Dimension {i}: {val:.4f}")

    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description='Analyze specific participants from dataset.csv')
    parser.add_argument('--csv_file', type=str, default='data/dataset.csv', help='Path to the dataset CSV file')
    parser.add_argument('--model_path', type=str, default='transformer_vae_static_model.pth', help='Path to load/save the model')
    parser.add_argument('--names', nargs='+', default=[], help='Names of participants to analyze')
    parser.add_argument('--indices', nargs='+', type=int, default=[], help='Indices of participants to analyze')
    parser.add_argument('--category_names', nargs='+', default=['R', 'I', 'A', 'S', 'E', 'C'], help='Names of the categories')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    df, participant_info, answers_data = load_dataset(args.csv_file)
    num_samples, num_questions = answers_data.shape
    print(f"Loaded {num_samples} participants with {num_questions} questions each")

    # Create question categories
    question_categories_data = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
    print(f"Question Categories (0-{config.NUM_CATEGORIES-1}): {question_categories_data.tolist()}")
    counts = torch.bincount(question_categories_data, minlength=config.NUM_CATEGORIES)
    print(f"Category Counts: {counts.tolist()}")

    # Load or train model
    model = load_or_train_model(answers_data, question_categories_data, device, args.model_path)

    # Analyze participants
    results = {}

    # By name
    if args.names:
        name_results = analyze_participants_by_name(
            model, args.names, answers_data, question_categories_data, participant_info, device
        )
        results.update(name_results)

    # By index
    for idx in args.indices:
        if 0 <= idx < num_samples:
            participant_name = participant_info.iloc[idx]['name']
            print(f"Analyzing participant at index {idx}: {participant_name}")
            participant_results = analyze_participant(
                model, idx, answers_data, question_categories_data, participant_info, device
            )
            results[participant_name] = participant_results
        else:
            print(f"Invalid participant index: {idx}")

    # If no specific participants are requested, analyze the first one
    if not args.names and not args.indices:
        print("No specific participants requested. Analyzing first participant.")
        participant_results = analyze_participant(
            model, 0, answers_data, question_categories_data, participant_info, device
        )
        results[participant_info.iloc[0]['name']] = participant_results

    # Visualize results
    visualize_participant_results(results, args.category_names)

    # Export results to CSV
    export_results_to_csv(results, args.category_names)

def export_results_to_csv(results, category_names):
    """
    Export analysis results to CSV files

    Parameters:
    results (dict): Analysis results
    category_names (list): Names of categories
    """
    # Prepare data for export
    export_data = []

    for participant_name, participant_results in results.items():
        row = {
            'Name': participant_name,
            'Student ID': participant_results['participant_id'],
            'Dominant Category (GT)': category_names[participant_results['dominant_category_gt']],
            'Dominant Category (Pred)': category_names[participant_results['dominant_category_pred']]
        }

        # Add static distribution values
        for i, cat_name in enumerate(category_names):
            row[f'{cat_name} (GT)'] = participant_results['static_distribution_gt'][0][i]
            row[f'{cat_name} (Pred)'] = participant_results['static_distribution_pred'][0][i]

        # Add latent space values
        for i, val in enumerate(participant_results['latent_mu'][0]):
            row[f'Latent Dim {i}'] = val

        export_data.append(row)

    # Convert to DataFrame and export
    results_df = pd.DataFrame(export_data)
    results_df.to_csv('participant_analysis_results.csv', index=False)
    print("Results exported to participant_analysis_results.csv")

if __name__ == "__main__":
    main()