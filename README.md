# Free Energy Principle and Catastrophic Forgetting

This project explores the relationship between **catastrophic forgetting** in neural networks and the **Free Energy Principle (FEP)** from neuroscience. We study how a model that learns a new task using **LoRA adapters** starts to forget its **original task**, and how this forgetting shows up as changes in “surprise” (free energy) over the original labels. On top of this baseline, we add several **regularizers** (L1, L2, and a Free-Energy regularizer) to see whether they can reduce catastrophic forgetting.

## 1. Project Overview

### Key Hypothesis

When a model is fine-tuned on a new task, it may gradually **forget the original task**.  
Viewed through the lens of free energy, as the model shifts toward the new task, the original task’s labels become more “surprising” under the model’s predictions. This should create measurable changes in the free-energy landscape between:

- **Task A** – original task the model was first trained on  
- **Task B** – new task learned through LoRA

We expect:

- Accuracy on **Task A** to drop as LoRA strength increases (catastrophic forgetting).
- Free energy / “surprise” on Task-A labels to **increase** as the model forgets Task A.
- Regularizers that constrain the LoRA adaptor (especially FE-based) may **slow down** this forgetting.

**Note:** All experiments, plots, and interpretation of results (catastrophic forgetting, free-energy trends, regularizer comparison, etc.) are documented in `analysis.ipynb`.

## 2. Key concepts

1. **Catastrophic Forgetting**

Catastrophic forgetting is the phenomenon that artificial neural networks tend to rapidly and drastically forget previously learned information when learning new information.


2. **Free Energy Principle (FEP)**

The Free Energy Principle states that any system resists change and seeks to minimize "surprise" (negative log-likelihood):

$$F = -\log p(x|\theta) = KL(q(z|x) || p(z|x)) + E_q[-\log p(x|z)]$$

Where:
- **F**: Free Energy (surprise/uncertainty)
- **x**: Observed data
- **z**: Latent representations
- **q**: Variational posterior (approximate inference)
- **p**: Prior/generative model

### In this Project (Simplified Version)

Free energy = how surprised the model is by the correct label.

Mathematically,

`𝐹 = − log 𝑝(true label ∣ 𝑥)`

If the model is confident and correct → 𝑝 close to 1 →  F is small.

If the model is unsure or wrong → 𝑝 is small → 𝐹 is large.


## 3. Methodology

1. **Train a base model on Task A**

   - Task A is the original objective the model should remember.
   - We train a small MLP (`BaseClassifier`) on Task A until it reaches high accuracy.

2. **Add LoRA adapters and train on Task B**

   - We freeze the base model and attach a **LoRA adapter** to the final layer (`LoRAClassifier`).
   - Task B acts like a new objective that can cause **catastrophic forgetting of Task A**.
   - LoRA has a scalar **strength** parameter:
     - `0.0` → pure Task-A base model  
     - `1.0` → fully LoRA-adapted Task-B model  

3. **Measure free energy (surprise) for both tasks**

   For logits `z` and labels `y`:

   - `F_A = - log p(y_A | x)`  (using Task-A labels)
   - `F_B = - log p(y_B | x)`  (using Task-B labels)

   where `p(y | x)` comes from `softmax(z)`.

   We track:

   - Free energy on Task-A labels as LoRA strength increases.
   - Free-energy **divergence** between Task A and Task B.

4. **Analyze and visualize free-energy patterns**

   - Plot **Task-A accuracy vs LoRA strength**.
   - Plot **Task-B accuracy vs LoRA strength**.
   - Plot **free-energy divergence vs LoRA strength**.
   - Visualize **histograms** of FE distributions at different strengths.

5. **Compare different regularizers on the LoRA adapter**

   This is the extension of the project:

   - **NONE** – LoRA is trained only with Task-B cross-entropy.  
   - **L2** – Penalize large weights by adding the sum of squared weights to the loss.
   - **L1** – Penalize the sum of absolute values of the weights.
   - **FE regularizer** – Adds a penalty on free energy for Task-A labels. 

   We ask:

   - How do these regularizers change catastrophic forgetting curves?
   - Does FE regularization better preserve Task-A performance or FE patterns while still learning Task B?


## 4. Synthetic Tasks and Models

### Synthetic data (Task A & Task B)

Implemented in `src/data.py`:

- Inputs: `X ∈ ℝ^{N×20}`
- **Task A**:
  - Depends on **features 0–4**.
  - Labels are a noisy linear threshold over these features.
- **Task B**:
  - Depends on **features 5–9**.
  - Uses the same `X` but a different weight vector and feature subset.
- Remaining features are random noise.
- A helper `set_global_seed` keeps experiments reproducible.

### Base model (Task A)

`src/models/base.py`:

- `BaseClassifier` – 3-layer feedforward network:
  - `fc1`: 20 → 64, ReLU  
  - `fc2`: 64 → 32, ReLU  
  - `fc3`: 32 → 2 (binary classification)  
- Trained on Task A using `ModelTrainer` (`src/training/trainer.py`).

### LoRA adapter (Task B)

`src/models/lora.py`:

- `LoRAClassifier` wraps a **frozen** base model.
- Adds a rank-`r` LoRA adapter to the last linear layer.
- Combines base and LoRA outputs via a learnable scalar `lora_strength` that we sweep between 0 and 1 at evaluation time.


### Regularized LoRA Training

All regularized LoRA experiments live in:

- `src/training/regularizers.py` — defines `train_lora_variant(...)`
- `src/fe/metrics.py` — utilities for computing free energy from logits

`train_lora_variant(...)`:

- Takes a **base Task-A model** and freezes all of its parameters.
- Wraps it with a **LoRA adapter** on the final layer (Task B head).
- Trains **only the LoRA parameters** on Task-B cross-entropy.
- Optionally adds one of three regularizers (L2, L1, or FE) to the loss.

---

### What is a Regularizer?

In machine learning, a **regularizer** is an extra term added to the loss function to shape how the model learns. Instead of only minimizing the task loss (here, cross-entropy on **Task B**), we add a penalty that:

- discourages certain types of solutions (e.g., very large weights)  
- encourages certain behaviours (e.g., not completely forgetting the original Task A)

In this project, we compare **three kinds of regularization** applied **only to the LoRA adapter** while the base model remains frozen.


### 1. L2 Regularization (Weight Decay)
- Penalize large weights by adding the sum of squared weights to the loss.

### 2. L1 Regularization (Sparsity)
- Penalize the sum of absolute values of the weights. L1 tends to push many weights toward exactly zero (sparsity).

### 3. Free-Energy (FE) Regularizer
- **Idea:** Instead of penalizing weight size, we penalize **high surprise on Task A labels**.
  
  - FE regularizer: `F_A = -log p(y_A | x)` (keep the old task unsurprising)  
  - Combined training objective (conceptually):  
    `Loss = Task B loss + lambda_fe * F_A`
    

  This regularizer works at the **behaviour level**. It directly discourages the model from making Task A labels highly surprising while it learns Task B. This is closer to the Free Energy Principle view of “minimizing surprise” than standard L1/L2 penalties.


## 5. Repository Structure

```text
.
├── analysis.ipynb         # Main notebook: runs experiments & plots
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data.py            # SyntheticDataGenerator, set_global_seed
    ├── fe/
    │   ├── __init__.py
    │   └── metrics.py     # compute_free_energy_from_logits, helpers
    ├── models/
    │   ├── __init__.py
    │   ├── base.py        # BaseClassifier
    │   └── lora.py        # LoRAClassifier
    └── training/
        ├── __init__.py
        ├── trainer.py     # ModelTrainer for Task A
        └── regularizers.py# train_lora_variant with NONE/L1/L2/FE

```

## 6. Installation

Conda (recommended)
```
# create and activate env
conda create -n fe-lora python=3.10 -y
conda activate fe-lora

# install pytorch 
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# install remaining deps
pip install -r requirements.txt
```

## 7. Running the Notebook

Start Jupyter:

jupyter notebook

Open analysis.ipynb and run cells in order:

- Section: Data & Base Model

    - Generates synthetic Task A/B data.

    - Trains the base MLP on Task A, reports accuracy.

- Section: LoRA + Catastrophic Forgetting

    - Trains LoRA adapters on Task B with different regularizers.

    - Produces:

        - Task-A & Task-B accuracy vs LoRA strength.

        - FE divergence vs LoRA strength.

        - FE histograms for different strengths.

- Section: Multi-seed Analysis

    - Runs multiple seeds.

    - Computes mean/std and 95% CIs for each regularizer.

All plots and tables used in the report / slides can be regenerated from this notebook.

## 8. Limitation

1. Toy data + toy tasks

Task A and B are synthetic, linearly-separable binary problems built from the same 20-dimensional Gaussian input. Only the feature subsets (1–5 vs 6–10) differ, so the setting is much simpler than real multi-task or continual-learning problems with complex correlations, language, vision, etc. This makes the results easy to interpret but limits how far we can generalize to real models.

2. Very small architecture

The base network is a 3-layer MLP (20 → 64 → 32 → 2) with one hidden representation. LoRA is only applied to the last layer, so it cannot capture deeper representational drift in large transformers. Conclusions about catastrophic forgetting / transfer may change for deeper networks or attention models.

3. Simplified “Free Energy” implementation

Free energy is approximated as the negative log-probability of the true label, and FE divergence is just a difference between two such averages. This ignores the full variational FEP formulation with explicit priors, latent variables, and KL terms between generative and recognition models. As a result, the link to the neuroscience FEP is conceptual rather than a faithful implementation.

4. Narrow regularization design and hyperparameter search

Only three regularizers are tested on the LoRA weights: L1, L2, and the FE-based penalty, with a small grid of λ and LoRA strengths. Other common techniques (weight decay on the base model, dropout, orthogonality constraints, etc.) are not compared. The number of random seeds is limited, so while we report CIs and p-values, statistical power is still modest.

5. LoRA configuration is restricted

LoRA rank r and where the adapter is attached are fixed choices (last layer only, single r). LoRA strength is applied as a simple scalar mixer. We do not explore more complex mixing schedules or training LoRA jointly under different strengths. This makes it unclear whether the FE regularizer would still behave similarly for different LoRA designs.

## 9. Conclusion

In this project, we study catastrophic forgetting in a simple continual-learning setup, where a LoRA-augmented classifier is first trained on Task A and then adapted to Task B. We compare four settings for training the LoRA adapter on Task B: no regularization, standard L1 and L2 weight regularization, and a free-energy (FE)–based regularizer that explicitly penalizes surprise on the original task. Without regularization, the model drifts toward Task B and forgets much of Task A. Our experiments show that while L1 freezes the base model, allowing little to no new learning. L2 and FE regularization both help stabilize performance, but the FE term offers the most balanced stability–plasticity trade-off where it preserves Task-A accuracy better while still allowing learning for Task B. Overall, this suggests that free-energy–inspired objectives are a potential promising way to control catastrophic forgetting.


## 10. Credits

This project was developed as an educational experiment on NeuroAI, catastrophic forgetting, and Free-Energy–inspired regularization. The initial idea for using the Free Energy Principle in this setting was inspired by Artem Kirsanov’s YouTube “A Universal Theory of Brain Function.” I would also like to thank Professor Weinan Sun and TA Ivan for their guidance, advice, and feedback throughout the development of this work.
