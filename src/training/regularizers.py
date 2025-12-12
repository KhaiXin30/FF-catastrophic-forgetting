
from copy import deepcopy
from typing import Tuple
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict
from ..models.lora import LoRAClassifier
from ..fe.metrics import compute_free_energy_from_logits




# ------------------------------------------------------------------
# Helper: dual-task dataloaders (Task A + Task B labels)
# ------------------------------------------------------------------
def prepare_dual_task_loaders(
    X,
    y_task_a,
    y_task_b,
    batch_size: int = 32,
    test_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train / test DataLoaders for the dual-task setting.

    Each batch will yield (X_batch, y_b_batch, y_a_batch) so that:
      - y_b is the main objective (Task B)
      - y_a is available for FE regularization (Task A).
    """
    X = np.asarray(X)
    y_task_a = np.asarray(y_task_a)
    y_task_b = np.asarray(y_task_b)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_split)
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = torch.FloatTensor(X[train_idx])
    y_a_train = torch.LongTensor(y_task_a[train_idx])
    y_b_train = torch.LongTensor(y_task_b[train_idx])

    X_test = torch.FloatTensor(X[test_idx])
    y_a_test = torch.LongTensor(y_task_a[test_idx])
    y_b_test = torch.LongTensor(y_task_b[test_idx])

    # IMPORTANT: order = (X, y_b, y_a) to match your original training loop
    train_ds = TensorDataset(X_train, y_b_train, y_a_train)
    test_ds = TensorDataset(X_test, y_b_test, y_a_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def evaluate_model_metrics(model,
                           X_tensor,
                           y_a_tensor,
                           y_b_tensor,
                           device) -> Dict[str, float]:
    """
    Evaluate Task A/B accuracy and FE_A / FE_B for a given model
    at its current LoRA strength.
    """
    model.eval()
    X_tensor = X_tensor.to(device)
    y_a_tensor = y_a_tensor.to(device)
    y_b_tensor = y_b_tensor.to(device)

    with torch.no_grad():
        logits, _ = model(X_tensor)
        preds = torch.argmax(logits, dim=1)

        acc_a = (preds == y_a_tensor).float().mean().item()
        acc_b = (preds == y_b_tensor).float().mean().item()

        fe_a = compute_free_energy_from_logits(logits, y_a_tensor)
        fe_b = compute_free_energy_from_logits(logits, y_b_tensor)

    fe_a_mean = fe_a.mean().item()
    fe_b_mean = fe_b.mean().item()
    fe_divergence = fe_b_mean - fe_a_mean

    return {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "fe_a": fe_a_mean,
        "fe_b": fe_b_mean,
        "fe_div": fe_divergence,
    }



def train_lora_variant(
    base_model: nn.Module,
    X: np.ndarray,
    y_task_a: np.ndarray,
    y_task_b: np.ndarray,
    regularizer: str = "none",   # "none", "l1", "l2", "fe"
    lambda_reg: float = 0.1,
    r: int = 4,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    test_split: float = 0.2,
    device: torch.device = None,
) -> Tuple[nn.Module, pd.DataFrame]:
    '''
    
        Train a LoRA-augmented classifier on Task B with an optional regularizer,
        then evaluate catastrophic forgetting across different LoRA strengths.

        This function:
        1. Deep-copies the given `base_model` and wraps it with `LoRAClassifier`.
        2. Freezes all base-model parameters and trains only the LoRA adapter
            (and its scalar strength, if applicable) on Task-B cross-entropy.
        3. Applies one of several regularizers to the LoRA parameters:
            - "none": no explicit regularization
            - "l2"  : L2 via optimizer weight decay
            - "l1"  : L1 penalty on LoRA weights (and scalar strength)
            - "fe"  : free-energy penalty using Task-A labels
        4. After training, sweeps over several LoRA strengths and computes
            metrics for Task A and Task B (accuracy and free energy).

        Args:
            base_model (nn.Module):
                Pre-trained base classifier to be LoRA-augmented. Its parameters
                are frozen during training.
            X (np.ndarray):
                Input feature matrix of shape (N, D), where N is the number of
                samples and D is the feature dimension.
            y_task_a (np.ndarray):
                Integer labels for Task A of shape (N,). Used for free-energy
                regularization and evaluation of catastrophic forgetting.
            y_task_b (np.ndarray):
                Integer labels for Task B of shape (N,). This is the main
                optimization target during training (cross-entropy loss).
            regularizer (str, optional):
                Type of regularization to apply on LoRA parameters. One of
                {"none", "l1", "l2", "fe"}. Defaults to "none".
            lambda_reg (float, optional):
                Strength/weight of the chosen regularizer. Interpreted as:
                - L2 weight decay coefficient if `regularizer == "l2"`
                - Multiplier on L1 norm if `regularizer == "l1"`
                - Multiplier on FE_A penalty if `regularizer == "fe"`.
                Defaults to 0.1.
            r (int, optional):
                LoRA rank used by `LoRAClassifier`. Controls the size of the
                low-rank adaptation matrices. Defaults to 4.
            lr (float, optional):
                Learning rate for the Adam optimizer. Defaults to 1e-3.
            epochs (int, optional):
                Number of training epochs for the LoRA adapter. Defaults to 50.
            batch_size (int, optional):
                Mini-batch size for training. Defaults to 32.
            test_split (float, optional):
                Fraction of data to reserve for the test set when building
                dual-task dataloaders. The current implementation uses only
                the training loader for optimization, but keeps the split
                consistent with the rest of the project. Defaults to 0.2.
            device (torch.device, optional):
                Torch device on which to run training and evaluation. If None,
                uses "cuda" when available, otherwise "cpu".

        Returns:
            Tuple[nn.Module, pd.DataFrame]:
                - trained_model: The LoRA-augmented model after training.
                (Base model is frozen; only LoRA adapter has been updated.)
                - metrics_df: A pandas DataFrame with one row per LoRA strength,
                containing:
                    - "acc_a": Task-A accuracy
                    - "acc_b": Task-B accuracy
                    - "fe_a": mean free energy on Task A
                    - "fe_b": mean free energy on Task B
                    - "fe_div": FE_B - FE_A (direction of adaptation)
                    - "regularizer": name of the regularizer used
                    - "lambda": value of `lambda_reg`
                    - "strength": LoRA strength used for that evaluation row
    
    '''
  

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fresh copy of base model
    base_copy = deepcopy(base_model).to(device)

    # Wrap with LoRA
    lora_model = LoRAClassifier(base_copy, r=r).to(device)

    # Freeze base model
    for p in lora_model.base_model.parameters():
        p.requires_grad = False

    # During training, use full LoRA strength and keep scalar fixed
    lora_model.lora_strength.data = torch.tensor(1.0, device=device)
    lora_model.lora_strength.requires_grad = False

    # Only train the LoRA A/B weights
    lora_params = list(lora_model.lora_fc3.parameters())

    # Optimizer (L2 via weight_decay if chosen)
    if regularizer == "l2":
        optimizer = torch.optim.Adam(lora_params, lr=lr, weight_decay=lambda_reg)
    else:
        optimizer = torch.optim.Adam(lora_params, lr=lr)

    criterion = nn.CrossEntropyLoss()

    # Dual-task loaders so batches include y_B and y_A
    train_loader, _ = prepare_dual_task_loaders(
        X, y_task_a, y_task_b, batch_size=batch_size, test_split=test_split
    )

    # ---- Training loop ----
    for epoch in range(epochs):
        lora_model.train()
        total_loss = 0.0

        for X_batch, y_b_batch, y_a_batch in train_loader:
            X_batch = X_batch.to(device)
            y_b_batch = y_b_batch.to(device)
            y_a_batch = y_a_batch.to(device)

            optimizer.zero_grad()
            logits, _ = lora_model(X_batch)

            probs = torch.softmax(logits, dim=1)
            probs = torch.clamp(probs, min=1e-10, max=1.0)

            loss = criterion(logits, y_b_batch)  # main Task-B CE

            # ---- Regularizers ----
            if regularizer == "fe":
                # FE penalty on Task A labels
                idx = torch.arange(len(X_batch), device=device)
                true_probs = probs[idx, y_a_batch]
                fe_a = -torch.log(true_probs)
                loss = loss + lambda_reg * fe_a.mean()

            elif regularizer == "l1":
                l1_penalty = 0.0
                for p in lora_model.lora_fc3.parameters():
                    l1_penalty = l1_penalty + p.abs().sum()
                l1_penalty = l1_penalty + lora_model.lora_strength.abs()
                loss = loss + lambda_reg * l1_penalty
            # "none" and "l2" are already handled

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[{regularizer.upper()}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # ---- Evaluation across LoRA strengths ----
    X_tensor_full = torch.FloatTensor(X)
    y_a_tensor_full = torch.LongTensor(y_task_a)
    y_b_tensor_full = torch.LongTensor(y_task_b)

    strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for s in strengths:
        lora_model.lora_strength.data = torch.tensor(float(s), device=device)
        metrics = evaluate_model_metrics(
            lora_model, X_tensor_full, y_a_tensor_full, y_b_tensor_full, device
        )
        metrics.update({
            "regularizer": regularizer,
            "lambda": lambda_reg,
            "strength": s,
        })
        results.append(metrics)

    return lora_model, pd.DataFrame(results)