# main.py
""" Main script to run the Transformer VAE + Static Head experiment """

import torch
import torch.optim as optim
import time

from matplotlib import pyplot as plt

# Import components from other files
import config
import data
import model
import train
import analysis

def visualize_category_distributions(target_distributions, predicted_distributions=None, category_names=None):
    """
    Visualize the target and (optionally) predicted category distributions

    Parameters:
    target_distributions (torch.Tensor): Target distributions from the data
    predicted_distributions (torch.Tensor, optional): Predicted distributions from the model
    category_names (list, optional): Names of the categories (e.g., ['R', 'I', 'A', 'S', 'E', 'C'])
    """
    if category_names is None:
        category_names = [f"Cat {i+1}" for i in range(target_distributions.shape[1])]

    num_samples = min(10, target_distributions.shape[0])  # Show up to 10 samples
    num_categories = target_distributions.shape[1]

    fig, axs = plt.subplots(num_samples, 1, figsize=(10, 2*num_samples), squeeze=False)

    # Calculate average distribution across all samples
    avg_target = target_distributions.mean(dim=0).numpy()

    if predicted_distributions is not None:
        avg_pred = predicted_distributions.mean(dim=0).numpy()

    # Plot average distribution at the top
    axs[0, 0].bar(category_names, avg_target, alpha=0.6, label='Target (Avg)')
    if predicted_distributions is not None:
        axs[0, 0].bar(category_names, avg_pred, alpha=0.6, label='Predicted (Avg)')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_title('Average Distribution Across All Participants')
    axs[0, 0].legend()

    # Plot individual distributions
    for i in range(1, num_samples):
        axs[i, 0].bar(category_names, target_distributions[i].numpy(), alpha=0.6, label='Target')
        if predicted_distributions is not None:
            axs[i, 0].bar(category_names, predicted_distributions[i].numpy(), alpha=0.6, label='Predicted')
        axs[i, 0].set_ylim(0, 1)
        axs[i, 0].set_title(f'Participant {i+1}')
        if i == 1:  # Only add legend to the first sample
            axs[i, 0].legend()

    plt.tight_layout()
    plt.savefig('category_distributions.png')
    plt.close()
    print("Visualization saved to 'category_distributions.png'")

if __name__ == "__main__":
    start_run_time = time.time()

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    answers_data, _ = data.load_and_process_csv_data("data/dataset.csv")
    question_categories_data = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
    config.NUM_SAMPLES = answers_data.shape[0]

    # Calculate target static distributions needed for the training loss
    target_static_distributions = data.calculate_target_static_distributions(
        answers_data, question_categories_data, answers_data.shape[0], config.NUM_CATEGORIES
    )
    # Create dataloaders (train includes targets, eval does not)
    train_loader, full_loader = data.prepare_dataloaders(
        answers_data, target_static_distributions, config.BATCH_SIZE
    )
    # Keep categories_tensor on CPU, model will move it to device as needed
    categories_tensor = question_categories_data # Used as input to model embedding


    # 3. Initialize Model and Optimizer
    personality_model = model.TransformerVAEStaticHead(
        num_questions=config.NUM_QUESTIONS,
        num_categories=config.NUM_CATEGORIES,
        embed_dim=config.EMBED_DIM,
        latent_dim=config.LATENT_DIM,
        hidden_dim_dec=config.HIDDEN_DIM_DEC,
        hidden_dim_static_head=config.HIDDEN_DIM_STATIC_HEAD,
        transformer_nhead=config.TRANSFORMER_NHEAD,
        transformer_nlayers=config.TRANSFORMER_NLAYERS,
        transformer_dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD
    ).to(device) # Move model to device

    optimizer = optim.Adam(personality_model.parameters(), lr=config.LEARNING_RATE)

    # 4. Train Model
    train.train_model(
        personality_model, train_loader, categories_tensor, optimizer, config, device
    )

    # 5. Run Analysis
    # Note: Pass original answers_data and question_categories_data for ground truth calculation in analysis
    analysis.run_analysis(
        personality_model, full_loader, answers_data, question_categories_data,
        categories_tensor, device, config
    )

    end_run_time = time.time()
    print(f"\n--- Total Run Finished (Duration: {end_run_time - start_run_time:.2f} sec) ---")
