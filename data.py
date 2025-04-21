""" Data generation and loading utilities """
import os

import pandas as pd
import torch
import torch.distributions as dist
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def generate_structured_synthetic_data(num_samples, num_questions, num_categories):
    """ Generates balanced structured synthetic data (copy from previous). """
    print("Generating STRUCTURED synthetic data (BALANCED Categories)...")
    if num_questions % num_categories != 0:
        print(f"Warning: num_questions ({num_questions}) is not perfectly divisible by num_categories ({num_categories}). Falling back to random.")
        if num_questions >= num_categories:
            q_cats = torch.cat([torch.arange(num_categories), torch.randint(0, num_categories, (num_questions - num_categories,))]).long()
            question_categories_data = q_cats[torch.randperm(num_questions)]
        else:
            question_categories_data = torch.randint(0, num_categories, (num_questions,)).long()
    else:
        questions_per_category = num_questions // num_categories
        print(f"Assigning {questions_per_category} questions per category.")
        balanced_cats = np.repeat(np.arange(num_categories), questions_per_category)
        np.random.shuffle(balanced_cats)
        question_categories_data = torch.from_numpy(balanced_cats).long()
    print(f"Question Categories (0-5): {question_categories_data.tolist()}")
    counts = torch.bincount(question_categories_data, minlength=num_categories)
    print(f"Category Counts: {counts.tolist()}")

    dirichlet_alpha = torch.ones(num_categories) * 0.7
    profile_distribution = dist.Dirichlet(dirichlet_alpha)
    weight_profile_strength = 5.0
    bias_yes_probability = -1.5
    answers_list = []
    for i in range(num_samples):
        latent_profile = profile_distribution.sample()
        participant_answers = torch.zeros(num_questions, dtype=torch.int)
        for q_idx in range(num_questions):
            category_index = question_categories_data[q_idx]
            profile_strength_for_category = latent_profile[category_index]
            logit = weight_profile_strength * profile_strength_for_category + bias_yes_probability
            p_yes = torch.sigmoid(logit); answer = (torch.rand(()) < p_yes).int()
            participant_answers[q_idx] = answer
        answers_list.append(participant_answers.unsqueeze(0))
    answers_data = torch.cat(answers_list, dim=0)
    print(f"Generated {num_samples} structured samples."); print(f"Overall 'Yes' percentage: {answers_data.float().mean().item()*100:.2f}%")
    return answers_data, question_categories_data

def calculate_target_static_distributions(answers_data, question_categories_data, num_samples, num_categories):
    """ Pre-calculates the static distribution for each sample (target for training). """
    print("Pre-calculating target static RIASEC distributions for training...")
    static_distributions_all = torch.zeros(num_samples, num_categories)
    for i in range(num_samples):
        answers_i = answers_data[i]
        yes_indices = torch.where(answers_i == 1)[0]
        num_yes = len(yes_indices)
        if num_yes > 0:
            categories_for_yes = question_categories_data[yes_indices]
            for cat_idx in range(num_categories):
                static_distributions_all[i, cat_idx] = (categories_for_yes == cat_idx).sum()
            static_distributions_all[i] /= num_yes
        # else: defaults to zeros
    print("Target static distributions calculated.")
    print(static_distributions_all[0:5])  # Print first 5 for verification
    return static_distributions_all

def prepare_dataloaders(answers_data, target_static_distributions, batch_size):
    """ Creates training and evaluation dataloaders. """
    # Training dataset needs both answers and target static distributions
    train_dataset = TensorDataset(answers_data, target_static_distributions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Evaluation dataset only needs answers as input to the model
    eval_dataset = TensorDataset(answers_data)
    full_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False) # No shuffle for consistent eval order

    return train_loader, full_loader

def load_and_process_csv_data(csv_file_path):
    """
    Load and process real data from CSV file

    Parameters:
    csv_file_path (str): Path to the CSV file containing survey responses
    num_categories (int): Number of categories (default is 6 for RIASEC)

    Returns:
    tuple: (answers_data, question_categories_data, participant_info)
    """
    # Load the CSV file
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Extract participant information
    participant_info = df[['name', 'student_id', 'timestamp']].copy()

    # Extract question responses (q1 to q30)
    question_columns = [col for col in df.columns if col.startswith('q')]
    answer_df = df[question_columns]

    # Convert to torch tensor
    answers_data = torch.tensor(answer_df.values, dtype=torch.int)
    num_samples, num_questions = answers_data.shape

    print(f"Loaded {num_samples} participants with {num_questions} questions each")

    # Calculate overall "Yes" percentage
    print(f"Overall 'Yes' percentage: {answers_data.float().mean().item()*100:.2f}%")

    return answers_data, participant_info

def merge_csv_files_and_process(csv_files, output_path='merged_data.csv', num_categories=6, batch_size=32):
    """
    Merge multiple CSV files and process the data for the Parseus model

    Parameters:
    csv_files (list): List of CSV file paths to merge
    output_path (str): Path to save the merged CSV file
    num_categories (int): Number of categories for classification
    batch_size (int): Batch size for dataloaders

    Returns:
    tuple: (train_loader, full_loader, answers_data, target_distributions, participant_info)
    """
    # Merge CSV files if multiple are provided
    if len(csv_files) > 1:
        print(f"Merging {len(csv_files)} CSV files...")
        df_list = []
        for file in csv_files:
            if not os.path.exists(file):
                print(f"Warning: File {file} does not exist and will be skipped")
                continue
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                print(f"Successfully read {file} with {df.shape[0]} rows")
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")

        if not df_list:
            raise ValueError("No valid CSV files could be read")

        # Concatenate all dataframes
        merged_df = pd.concat(df_list, ignore_index=True)

        # Save the merged dataframe to a CSV file
        merged_df.to_csv(output_path, index=False)
        print(f"Merged data saved to {output_path}")

        # Process the merged data
        answers_data, question_categories_data, participant_info = load_and_process_csv_data(output_path, num_categories)
    else:
        # Process the single CSV file
        answers_data, question_categories_data, participant_info = load_and_process_csv_data(csv_files[0], num_categories)

    # Calculate target distributions
    num_samples = answers_data.shape[0]
    target_distributions = calculate_target_static_distributions(
        answers_data,
        question_categories_data,
        num_samples,
        num_categories
    )

    # Prepare dataloaders for training and evaluation
    train_loader, full_loader = prepare_dataloaders(
        answers_data,
        target_distributions,
        batch_size
    )

    return train_loader, full_loader, answers_data, target_distributions, participant_info, question_categories_data

