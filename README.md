# Efficient-Memo: Incremental Learning with EfficientNet

Efficient-Memo is a PyTorch-based deep learning project that uses EfficientNet for incremental learning on an agricultural pest dataset. The project trains a model to classify images of pests into multiple categories while supporting incremental learning to adapt to new classes.

## Features
- **Incremental Learning**: Train the model in stages, adding new classes incrementally.
- **EfficientNet Backbone**: Leverages the EfficientNet architecture for feature extraction.
- **Dataset Handling**: Supports agricultural pest datasets structured using `train`, `val`, and `test` splits.
- **Customizable**: Easily adjust the model, learning rate, and dataset paths.

---

## Table of Contents
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

---

## Installation

### Prerequisites
- Python 3.8 or above
- PyTorch and torchvision
- Additional libraries: `efficientnet-pytorch`, `numpy`, `scikit-learn`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Efficient-Memo.git
   cd Efficient-Memo

Create a virtual environment:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt
