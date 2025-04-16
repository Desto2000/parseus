# data.py
""" Data generation and loading utilities """

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
