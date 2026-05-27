# ECG Generator and Classifier — Repository Context for Claude Code

## Purpose of This Document

This document exists to give an AI coding agent (Claude Code) deep context about the repository, the goals of the project, the intended architecture, the reasoning behind major technical choices, and the expected development philosophy.

The goal is NOT just to make code changes mechanically.
The goal is to help the agent understand:

* what this project is trying to achieve
* why certain architectural choices were made
* what tradeoffs matter
* how the repository should evolve
* how to refactor safely
* what kinds of abstractions are desirable
* what assumptions should NOT be broken

This repository is a research-oriented ECG deep learning system focused on:

1. Multi-lead ECG classification
2. Synthetic ECG generation / augmentation
3. Explainability and interpretability
4. Reproducible experimentation
5. Building toward clinically meaningful AI systems

The repository may currently contain:

* experimental code
* duplicate notebooks
* partially working pipelines
* rapidly prototyped modules
* inconsistent naming
* unused files
* temporary debugging artifacts
* mixed research and production code

The purpose of future cleanup/refactoring is to:

* improve maintainability
* improve modularity
* improve reproducibility
* improve readability
* improve experiment tracking
* reduce technical debt
* preserve research flexibility

WITHOUT destroying experimental agility.

---

# High-Level Project Vision

The long-term vision is to create a robust ECG AI research framework capable of:

* ECG classification
* ECG generation
* ECG reconstruction
* ECG denoising
* explainability
* representation learning
* potentially multimodal clinical modeling later

The repository is intended to evolve into a modular ECG research platform rather than a single throwaway model implementation.

The project values:

* clarity
* reproducibility
* interpretability
* modularity
* experiment iteration speed
* correctness over cleverness

The project does NOT value:

* unnecessary abstraction
* overengineering
* premature optimization
* opaque code
* hyper-compressed one-liners
* “smart” but unreadable implementations

---

# Domain Context

## ECG Basics

ECG data consists of electrical cardiac signals measured over time.

Typical setup:

* 12 leads
* sampled over time
* continuous 1D signals

Example shape:

12 × 1000

meaning:

* 12 leads
* 1000 timesteps

The project primarily works with multi-lead ECG recordings.

Key properties of ECG data:

* highly structured temporal signals
* noisy
* clinically imbalanced
* subtle inter-class variation
* strong temporal dependencies
* lead-specific information
* high inter-patient variability

This means the repository must support:

* temporal modeling
* lead-aware modeling
* class imbalance handling
* robust preprocessing
* explainability

---

# Current Research Direction

The repository currently focuses heavily on:

## ECG Classification

Primary dataset:

* PTB-XL

Potential future datasets:

* MIT-BIH
* Chapman
* PhysioNet datasets
* custom datasets

Classification task:

* multi-class ECG diagnosis
* potentially multi-label later

Current approach:

* CNN-based architectures
* attention mechanisms
* explainability with Integrated Gradients

Evaluation emphasis:

* Macro-F1
* imbalance-aware metrics
* confusion matrix analysis
* interpretability

---

# Core Technical Philosophy

## 1. Reproducibility Matters

All experiments should ideally be reproducible.

Desired future structure:

* config-driven experiments
* deterministic seeds
* centralized hyperparameters
* structured experiment folders
* checkpoint management
* logging
* metrics persistence

Avoid:

* hardcoded paths
* hidden constants
* notebook-only logic
* stateful hidden assumptions

---

## 2. Interpretability Is a First-Class Concern

This project explicitly values explainability.

The goal is NOT merely maximizing accuracy.

Interpretability components may include:

* Integrated Gradients
* saliency methods
* attention visualization
* lead-wise attribution
* temporal attribution
* attribution aggregation

Refactors should preserve explainability compatibility.

Avoid architectural changes that:

* make attribution impossible
* obscure signal flow
* destroy traceability

---

## 3. Research Flexibility Is Important

The repository should support rapid experimentation.

That means:

* modular models
* swappable losses
* configurable datasets
* easy experimentation
* minimal friction for new ideas

Avoid overly rigid frameworks.

The codebase should feel like:

"structured research code"

NOT:

"enterprise backend software"

---

## 4. Cleanliness Still Matters

Even though this is research code, cleanup is important.

Desired characteristics:

* readable
* modular
* discoverable
* documented
* minimally duplicated
* logically organized

The repository should eventually become easy for:

* collaborators
* researchers
* future maintainers
* AI coding agents

---

# Likely Repository Components

The repository may contain some or all of the following:

## Data Processing

Potential responsibilities:

* loading PTB-XL
* waveform normalization
* lead handling
* segmentation
* sliding windows
* label mapping
* train/val/test splitting
* augmentation
* filtering

Potential cleanup goals:

* centralize preprocessing
* remove duplicated transforms
* unify dataset APIs

---

## Models

Potential model families:

* 1D CNNs
* attention CNNs
* residual CNNs
* transformers (future)
* hybrid architectures
* generative models

Desired architecture qualities:

* modular
* composable
* readable
* debuggable

Avoid:

* deeply entangled forward passes
* hidden tensor reshaping logic
* implicit assumptions

Tensor shapes should be easy to trace.

---

## Training

Training logic should ideally be separated from:

* model definitions
* evaluation
* explainability
* data loading

Desired future structure:

* train.py
* evaluate.py
* configs/
* utils/
* trainers/

Potential future additions:

* mixed precision
* distributed training
* experiment tracking
* early stopping
* scheduler support

---

## Explainability

This is an important subsystem.

Possible explainability features:

* Integrated Gradients
* saliency maps
* attribution heatmaps
* lead importance
* temporal visualization

Desired properties:

* model-agnostic where possible
* reusable APIs
* visualization utilities
* minimal duplication

---

## ECG Generation

The repository may include or later include:

* GANs
* VAEs
* diffusion models
* waveform synthesis
* latent interpolation
* augmentation pipelines

Important:

Generated ECGs should aim for:

* physiologically plausible morphology
* stable waveform structure
* realistic lead relationships

Future cleanup should keep generative modeling extensibility in mind.

---

# Preferred Repository Organization

The repository does NOT necessarily already follow this structure.
This is the desired direction.

Suggested structure:

```text
project_root/
│
├── configs/
├── data/
├── datasets/
├── models/
├── training/
├── evaluation/
├── explainability/
├── generation/
├── utils/
├── notebooks/
├── scripts/
├── experiments/
├── checkpoints/
├── docs/
└── tests/
```

---

# Important Cleanup Goals

Claude Code should prioritize the following:

## 1. Remove Dead Code Carefully

Potential dead code exists.

However:

Some experimental code may look unused but still contains valuable ideas.

Before deleting:

* check imports
* check notebooks
* check experimental references
* check model variants

Prefer:

* archiving
* moving to experimental/

instead of aggressive deletion.

---

## 2. Reduce Duplication

The repository may contain:

* repeated preprocessing
* duplicated utility functions
* multiple versions of training loops
* copied notebook logic

Goal:

centralize shared logic.

---

## 3. Improve Naming

Bad names create major maintenance problems.

Prefer names that communicate:

* purpose
* tensor shape expectations
* domain meaning

Avoid vague names like:

* temp
* thing
* x1
* model2
* final_final

---

## 4. Improve Configuration Management

Reduce:

* hardcoded constants
* magic numbers
* hidden hyperparameters

Prefer:

* config files
* dataclasses
* centralized constants

---

## 5. Improve Logging and Experiment Tracking

Desired future support:

* TensorBoard
* Weights & Biases
* CSV logs
* checkpoint metadata
* reproducible runs

---

# Coding Style Preferences

## General Style

Prefer:

* readability
* explicitness
* moderate verbosity
* simple control flow
* comments where useful

Avoid:

* hyper-condensed code
* clever hacks
* unnecessary abstractions
* hidden side effects

---

## Functions

Functions should ideally:

* do one thing
* have clear inputs/outputs
* avoid implicit state
* be easy to test

---

## Classes

Use classes when they provide:

* meaningful abstraction
* reusable state
* cleaner interfaces

Do NOT introduce classes unnecessarily.

---

## Tensor Handling

Tensor dimensions should be extremely clear.

ECG projects become difficult to debug when tensor semantics are unclear.

Good practice:

```python
# shape: [batch, leads, time]
```

throughout the codebase.

---

# Performance Philosophy

Correctness and clarity are more important than micro-optimizations.

However, avoid:

* obvious inefficiencies
* repeated preprocessing
* unnecessary CPU/GPU transfers
* duplicated inference passes

Potential future optimization areas:

* dataloading
* mixed precision
* caching
* vectorization

---

# Documentation Expectations

Claude Code should improve documentation where appropriate.

Important targets:

* README clarity
* setup instructions
* training instructions
* inference examples
* explainability usage
* architecture descriptions
* dataset preparation

Code comments should explain:

* WHY something exists

not merely:

* WHAT the syntax does

---

# Notebook Philosophy

The repository may contain many notebooks.

Notebooks are acceptable for:

* experimentation
* visualization
* rapid iteration
* exploratory analysis

However:

Core reusable logic should eventually migrate into Python modules.

Avoid duplicating large chunks of code across notebooks.

---

# Evaluation Philosophy

The project prioritizes robust evaluation.

Preferred metrics:

* Macro-F1
* precision/recall
* confusion matrices
* class-wise analysis

Why Macro-F1 matters:

ECG datasets are highly imbalanced.

Raw accuracy can be misleading.

The repository should continue emphasizing imbalance-aware evaluation.

---

# Explainability Philosophy

Important conceptual boundary:

Explainability is intended to help:

* audit model behavior
* debug models
* analyze learned patterns

It is NOT intended to claim:

* medical certainty
* pathology localization
* clinical decision authority

Documentation and code comments should preserve this distinction.

---

# Potential Future Directions

Possible future work includes:

* transformers for ECG
* self-supervised learning
* contrastive learning
* diffusion ECG generation
* multimodal learning
* ECG-text models
* beat-level modeling
* clinician evaluation
* uncertainty estimation
* deployment tooling

The repository should remain flexible enough to support future research.

---

# Refactoring Guidance

When refactoring:

## Prefer Incremental Refactors

Avoid massive rewrites unless necessary.

Preserve:

* existing functionality
* experiment reproducibility
* model behavior

---

## Avoid Breaking Research Workflows

Researchers often rely on:

* notebooks
* scripts
* partially experimental pipelines

Refactors should not aggressively break workflows.

---

## Preserve Interpretability Compatibility

This is critical.

Attribution methods often depend on:

* differentiability
* accessible activations
* predictable forward passes

Avoid architectural changes that break attribution support.

---

# Suggested Immediate Improvements

Potential high-value cleanup areas:

1. Repository structure cleanup
2. Remove duplicated preprocessing
3. Centralize configs
4. Create reusable training utilities
5. Standardize model interfaces
6. Improve README/setup docs
7. Add consistent logging
8. Add typing where useful
9. Improve checkpoint organization
10. Separate experiments from reusable code

---

# Important Assumptions

Claude Code should assume:

* this is an active research repository
* experimentation speed matters
* explainability matters deeply
* readability matters
* future extensibility matters
* clinical sensitivity exists
* reproducibility is important

Claude Code should NOT assume:

* the repository is production-ready
* all code is finalized
* all notebooks are obsolete
* aggressive abstraction is desired
* enterprise software patterns are appropriate everywhere

---

# Recommended AI-Agent Behavior

When making changes:

1. First understand the architecture
2. Trace tensor flows carefully
3. Preserve experimental flexibility
4. Prefer modular improvements
5. Improve readability where possible
6. Avoid unnecessary rewrites
7. Add documentation when useful
8. Reduce duplication carefully
9. Preserve explainability compatibility
10. Avoid deleting potentially valuable experiments blindly

---

# Final Notes

This repository represents a long-term ECG AI research effort.

The project is not just about:

"training a classifier"

It is about building:

* a flexible ECG research framework
* interpretable medical AI systems
* reproducible experimentation pipelines
* foundations for future clinical research

All future modifications should respect:

* scientific clarity
* engineering maintainability
* interpretability
* extensibility
* research velocity

The ideal outcome is a repository that remains:

* easy to experiment with
* easy to understand
* easy to extend
* easy to debug
* and robust enough for serious ECG AI research.
