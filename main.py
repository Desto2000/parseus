# main.py
""" Main script to run the Transformer VAE + Static Head experiment """

import torch
import torch.optim as optim
import time

# Import components from other files
import config
import data
import model
import train
import analysis

if __name__ == "__main__":
    start_run_time = time.time()

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    answers_data, question_categories_data = data.generate_structured_synthetic_data(
        config.NUM_SAMPLES, config.NUM_QUESTIONS, config.NUM_CATEGORIES
    )
    # Calculate target static distributions needed for the training loss
    target_static_distributions = data.calculate_target_static_distributions(
        answers_data, question_categories_data, config.NUM_SAMPLES, config.NUM_CATEGORIES
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
