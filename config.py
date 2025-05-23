""" Hyperparameters and configuration settings """

# Data parameters
NUM_SAMPLES = 1000
NUM_QUESTIONS = 30
NUM_CATEGORIES = 6  # RIASEC

# Model parameters
EMBED_DIM = 64
LATENT_DIM = 6      # Dimension of the latent space (z) - Should match NUM_CATEGORIES ideally
HIDDEN_DIM_DEC = 64 # Hidden dimension for decoder MLP layers
HIDDEN_DIM_STATIC_HEAD = 32 # Hidden dimension for static prediction head

# --- Transformer Specific Hyperparameters ---
TRANSFORMER_NHEAD = 8       # Number of attention heads
TRANSFORMER_NLAYERS = 4     # Number of encoder layers
TRANSFORMER_DIM_FEEDFORWARD = 128 # Dimension of feedforward network in Transformer

# Training parameters
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100    # Adjust as needed
BATCH_SIZE = 16
BETA = 1.0         # Weight for KL divergence loss (VAE regularization)
GAMMA = 1.0        # Weight for static distribution prediction loss - TUNABLE
ALPHA = 1.0        # Weight for reconstruction loss (BCE) - TUNABLE

# Analysis parameters
N_CLUSTERS_EVAL = NUM_CATEGORIES # Number of clusters for K-Means eval on latent space
NUM_VECTORS_COMPARE = 10       # How many samples for side-by-side vector comparison

# --- File names ---
# Model Save Path
MODEL_SAVE_PATH = "out/transformer_vae_static_model.pth"

# File names for saving results
PLOT_FILENAME_2D = "out/transformer_vae_static_analysis_2d.png"
PLOT_FILENAME_2D2 = "out/transformer_vae_static_analysis_2d_svd.png"
PLOT_FILENAME_3D_LATENT = "out/transformer_vae_static_latent_3d.html"
PLOT_FILENAME_3D_STATIC = "out/transformer_vae_static_static_dist_3d.html" # Plotting static dists for comparison
PLOT_FILENAME_CORR_LATENT = "out/transformer_vae_static_latent_correlation.png"
PLOT_FILENAME_COSINE_CENTROIDS = "out/transformer_vae_static_kmeans_centroid_cosine.png"

PLOT_FILENAME_PR_CURVE = "out/pr_curve.png"
PLOT_FILENAME_ROC_CURVE = "out/roc_curve.png"