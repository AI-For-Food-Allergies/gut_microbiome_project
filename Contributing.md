# Contribution guideline

This document provides an analysis of the current code structure in the `modularize-training` branch of the `AI-For-Food-Allergies/gut_microbiome_project` repository and outlines the necessary interventions as defined by the open GitHub issues.

## Table of Contents

1. [Current Code Structure](#1-current-code-structure)

2. [Open issues](#2-open-issues)
3. [Development Roadmap](#3-development-roadmap)

4. [How to Contribute](#4-how-to-contribute)


## 1. Current Code Structure

### 1.1. High-Level Overview

The project is structured around the core components of a machine learning workflow, with dedicated directories and files for each step.

| Component | Directory/File | Purpose |
| :--- | :--- | :--- |
| **Data Preparation** | `data_preprocessing/` | Contains logic for cleaning, transforming, and preparing raw data. |
| **Model Components** | `modules/` | Intended to house reusable classes and functions for the model architecture (e.g., `MicrobiomeTransformer`). |
| **Training/Execution** | `train.py`, `main.py` | The main entry points for running the training and overall pipeline. |
| **Evaluation** | `evaluation/` | Intended to house scripts or modules for model performance assessment. |
| **Configuration** | `pyproject.toml` | Project dependencies and packaging. |
| **Utilities** | `data_loading.py` | Contains functions for loading data, which is a key part of the workflow. |
| **Examples/Legacy** | `example_scripts/` | Example scripts training a classifier based on MicrobiomeTransformer(`predict_milk.py`, `predict_hla.py`) |

### 1.2. Modular approach


*   **Separation of Concerns:** The creation of `data_preprocessing/`, `modules/`, and `evaluation/` clearly separates the data, model, and assessment logic.
*   **Centralized Execution:** `main.py` and `train.py` serve as clean entry points, abstracting the complexity of the underlying modules.
*   **Data Handling:** Use `data_loading.py` to separate the I/O logic from the core ML algorithms.

## 2. Open issues

| Issue ID | Title | Description of Intervention |
| :--- | :--- | :--- |
| \#8 | **Implement data loading** | This task involves refactoring the data loading logic into a dedicated, robust module that handles all I/O, ID alignment, and data structure creation, likely consolidating logic from `data_loading.py` and `utils.py`. |
| \#7 | **Implement training script** | This task requires finalizing and centralizing the training loop logic within `train.py`, ensuring it correctly imports the necessary data loader and model modules to execute the full training process (e.g., cross-validation, hyperparameter tuning). |
| \#6 | **Implement modules** | This task is to complete the implementation of all reusable model components (e.g., the `MicrobiomeTransformer` wrapper, feature extraction logic) and place them within the `modules/` directory, decoupling the model logic from the training script. |
| \#5 | **Implement evaluation script** | This task is to develop a dedicated script or module (e.g., `evaluation/evaluate.py`) that takes a trained model and test data, and produces the required metrics and visualizations, replacing the evaluation logic currently embedded in the legacy scripts. |

## 3. Development Roadmap
### Phase 1: Core Module Implementation

The first step is to build the reusable components that the main scripts will rely on.

1.  **Address Issue \#6: Implement modules**
    *   Define and implement the core model classes (e.g., `MicrobiomeModel`, `FeatureExtractor`) in the `modules/` directory.
    *   Ensure these modules are clean, well-documented, and only handle model-related logic.

2.  **Address Issue \#8: Implement data loading**
    *   Create a dedicated data module (e.g., `data_preprocessing/data_loader.py`).
    *   Move all data-related functions (SRA ↔ MicrobeAtlas ↔ DIABIMMUNE mappings, embedding loading, sample vector building) from `data_loading.py` and `utils.py` into this new module.
    *   The goal is to have a single, clean interface for retrieving processed data ready for training.

### Phase 2: Pipeline Integration

Once the core components are modularized, they can be integrated into the main execution scripts.

3.  **Address Issue \#7: Implement training script**
    *   Refine `train.py` to import the new data loader and model modules.
    *   Implement the final, clean training loop, including the 5-fold Stratified CV and cohort balancing logic.
    *   Ensure the script saves the trained model artifacts.

4.  **Address Issue \#5: Implement evaluation script**
    *   Create `evaluation/evaluate.py`.
    *   Implement the logic to load a trained model, load the test data using the new data loader, compute metrics, and generate the required plots (`milk_cm.png`, `fla_cm.png`).
    *   This script should be runnable independently to assess any trained model.

### Phase 3: Cleanup and Finalization

5.  **Configuration Refinement:** Review `config.yaml` and ensure all new modules and scripts correctly reference the configuration parameters.
6.  **Documentation:** Update the main `README.md` to reflect the new modular structure and provide clear instructions on how to run the `train.py` and `evaluation/evaluate.py` scripts.
7.  **Deprecation:** Remove or clearly mark the legacy/notebook-like scripts in `example_scripts/` to prevent confusion.

## 4. How to Contribute

We welcome contributions to this project! Follow these steps to contribute effectively:

### 4.1. Getting Started

1.  **Fork the Repository:** Create your own fork of the project on GitHub.
2.  **Clone Your Fork:** Clone the repository to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/gut_microbiome_project.git
    cd gut_microbiome_project
    ```
3.  **Set Up the Environment:** Install the project dependencies using your preferred package manager:
    ```bash
    pip install -e .
    ```
    or if using `uv`:
    ```bash
    uv sync
    ```

### 4.2. Making Changes

1.  **Create a Branch:** Always create a new branch for your work. Use a descriptive name:
    ```bash
    git checkout -b feature/your-feature-name
    ```
    or
    ```bash
    git checkout -b fix/issue-number-description
    ```

2.  **Follow the Development Roadmap:** Refer to Section 3 to understand the current priorities and ensure your contribution aligns with the project goals.

3.  **Write Clean Code:**

    *   Keep functions focused and modular.
    *   Use type hints where appropriate.

4.  **Test Your Changes:** Ensure your code works as expected and doesn't break existing functionality.

### 4.3. Submitting Your Contribution

1.  **Commit Your Changes:** Write clear, concise commit messages:
    ```bash
    git add .
    git commit -m "Add feature: brief description of what you did"
    ```

2.  **Push to Your Fork:**
    ```bash
    git push origin feature/your-feature-name
    ```

3.  **Open a Pull Request:**
    *   Navigate to the original repository on GitHub.
    *   Click "New Pull Request" and select your branch.
    *   Provide a clear description of your changes, referencing any related issues (e.g., "Closes #8").
    *   Wait for review and address any feedback.

### 4.4. Contribution Guidelines

*   **Focus on Open Issues:** Check the issues listed in Section 2 and prioritize work that addresses them.
*   **Keep PRs Focused:** Each pull request should address a single issue or feature.
*   **Document Your Work:** Update relevant documentation (README, docstrings) as needed.
*   **Be Responsive:** Respond to code review comments and be open to suggestions.
*   **Respect the Structure:** Follow the modular architecture outlined in this document.

### 4.5. Need Help?

If you have questions or need guidance:
*   Open an issue on GitHub to discuss your proposed changes.
*   Reach out to the maintainers for clarification on architectural decisions.
*   Review the `README.md` and `CONFIG_GUIDE.md` for additional context.

Thank you for contributing to the gut microbiome project!