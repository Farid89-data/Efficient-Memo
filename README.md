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
   git clone https://github.com/Farid89-data/Efficient-Memo.git
   cd Efficient-Memo
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. Install dependencies:

   ```bash
   pip install -r requirements.txt

## Folder Structure

Below is the expected folder structure for the project:
~~~text
Efficient-Memo/
├── dataset/
│   ├── kaggle_data/
│   │   ├── train/
│   │   │   ├── class1/
│   │   │   ├── class2/
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── class1/
│   │   │   ├── class2/
│   │   │   └── ...
│   │   └── val/ (optional)
├── main.py
├── requirements.txt
└── README.md
~~~
## Usage
## Dataset Preparation

Ensure your dataset is structured as follows:
~~~text
kaggle_data/
├── train/
│   ├── ants/
│   ├── bees/
│   └── ...
├── test/
│   ├── ants/
│   ├── bees/
│   └── ...
~~~
2. Update Dataset Path

In `main.py`, set the `data_dir` variable to the path of your dataset:
   ```
data_dir = "C:/Users/mehr110/PycharmProjects/Efficient-Memo/dataset/aggle_data"
```
3. Run the Training Script

Run the script to train the model:
```
python main.py
```

### Model Architecture

Efficient-Memo uses the EfficientNet backbone, with additional layers for incremental learning. The model architecture includes:

    1- Generalized Layers : Pretrained EfficientNet backbone layers.
    2- Specialized Layers: Custom layers for feature extraction.
    3- Fully Connected Layer: Final classification layer for the number of classes.


 ###   Results
# Example Output

The training script prints epoch losses and validation accuracy:
~~~
Epoch 1/10, Loss: 0.5432, Val Acc: 0.8543
Epoch 2/10, Loss: 0.4321, Val Acc: 0.9123
...
Best Validation Accuracy: 0.9123
~~~

### License

This project is licensed under the MIT License.
