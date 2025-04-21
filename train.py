""" Training loop and loss function definition """
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
import time


def loss_function(recon_logits, answers, mu, logvar,
                  predicted_static_dist_logits, target_static_dist,
                  beta, gamma, alpha):
    """
    Calculates the combined VAE loss: BCE + KLD + Static MSE + Entropy.
    Args:
        recon_logits: Raw output logits from the decoder.
        answers: Ground truth answers (binary).
        mu: Latent mean.
        logvar: Latent log variance.
        predicted_static_dist_logits: Raw output logits from the static prediction head.
        target_static_dist: Ground truth static distributions (normalized).
        beta: Weight for KLD loss.
        gamma: Weight for Static MSE loss.
        alpha: Weight for BCE loss.
    Returns:
        Tuple: (total_loss, BCE, KLD, Static_MSE)
    """
    # Reconstruction Loss
    BCE = F.binary_cross_entropy_with_logits(recon_logits, answers.float(), reduction='sum')

    # KL Divergence Loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    predicted_probs = F.softmax(predicted_static_dist_logits, dim=1)
    static_KLD = F.kl_div(torch.log(predicted_probs), target_static_dist, reduction='sum')

    entropy_loss = F.cross_entropy(predicted_probs, target_static_dist, reduction='sum')

    # Total Loss
    total_loss = alpha * BCE + beta * KLD + gamma * entropy_loss

    return total_loss, BCE, KLD, static_KLD, entropy_loss


def train_model(model, train_loader, categories_tensor, optimizer, config, device):
    """ Trains the TransformerVAEStaticHead model. """
    model.train()
    model.to(device)
    print(f"\n--- Starting Training (Device: {device}) ---")
    start_time = time.time()

    num_epochs = config.NUM_EPOCHS
    beta = config.BETA
    gamma = config.GAMMA
    alpha = config.ALPHA

    for epoch in range(num_epochs):
        total_loss_epoch, total_bce_epoch, total_kld_epoch, total_static_epoch, total_entropy_epoch = (0.0, 0.0, 0.0,
                                                                                                       0.0, 0.0)

        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, (answers_batch, static_dist_batch) in enumerate(pbar):
            answers_batch = answers_batch.to(device)
            static_dist_batch = static_dist_batch.to(device) # Target static distributions

            # Forward pass
            recon_logits, mu, logvar, predicted_static_logits = model(answers_batch, categories_tensor)

            # Calculate loss
            loss, bce, kld, static_kld, entropy = loss_function(
                recon_logits, answers_batch, mu, logvar,
                predicted_static_logits, static_dist_batch,
                beta, gamma, alpha
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping if needed
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses (batch sums)
            total_loss_epoch += loss.item()
            total_bce_epoch += bce.item()
            total_kld_epoch += kld.item()
            total_static_epoch += static_kld.item()
            total_entropy_epoch += entropy.item()
            # total_kll_epoch += kll.item()

            # Update progress bar description (optional)
            pbar.set_postfix({
                'Loss': f"{loss.item()/len(answers_batch):.4f}",
                'BCE': f"{bce.item()/len(answers_batch):.4f}",
                'KLD': f"{kld.item()/len(answers_batch):.4f}",
                'Static': f"{static_kld.item()/len(answers_batch):.4f}",
                'Entropy': f"{entropy.item()/len(answers_batch):.4f}",
                # 'KLL': f"{kll.item()/len(answers_batch):.4f}"
            })

        # Calculate average losses per sample for the epoch
        num_samples_in_loader = len(train_loader.dataset)
        avg_loss = total_loss_epoch / num_samples_in_loader
        avg_bce = total_bce_epoch / num_samples_in_loader
        avg_kld = total_kld_epoch / num_samples_in_loader
        avg_static = total_static_epoch / num_samples_in_loader
        avg_entropy = total_entropy_epoch / num_samples_in_loader
        # avg_kll = total_kll_epoch / num_samples_in_loader
        print(f" - "
              f"Avg Loss: {avg_loss:.4f}, "
              f"BCE: {avg_bce:.4f}, "
              f"KLD: {avg_kld:.4f}, "
              f"Static KLD: {avg_static:.4f}, "
              f"Entropy: {avg_entropy:.4f}, "
              # f"KLL: {avg_kll:.4f}"
              )

    end_time = time.time()
    print(f"--- Training Finished (Duration: {end_time - start_time:.2f} sec) ---")
