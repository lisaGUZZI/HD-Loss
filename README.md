# Regional Hausdorff Distance Losses for Medical Image Segmentation

This repository contains:
- a modified version of the nnU-Net v2 framework to include novel regional Hausdorff distance losses for medical image segmentation, as proposed in the paper "Regional Hausdorff Distance Losses for Medical Image Segmentation".
- The dataset splits to perform the results showed in the paper on the Pancreas dataset.
- The code for the differentiable Unsigned, Signed and Positive Distance Transforms.


## New Loss Functions

We introduce three new loss functions:

-   **LH (Hausdorff Loss)**: A differentiable version of the Hausdorff Distance.
-   **LAH (Averaged Hausdorff Distance Loss)**: A differentiable version of the Modified Hausdorff Distance .
-   **LAHsym (Symmetric Averaged Hausdorff Distance Loss)**: A variant of the LAH loss.

These losses can be used on their own or combined with the standard Cross-Entropy loss.

The implementations can be found in `nnunetv2/training/loss/`. The corresponding trainers are in `nnunetv2/training/nnUNetTrainer/variants/loss/`.

## Installation

This project is based on nnU-Net v2. It is recommended to follow the official nnU-Net v2 installation guide and then replace the code with the one from this repository.

1.  **Clone this repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    pip install -e .
    ```


2.  **Set up nnU-Net paths:**
    nnU-Net requires several environment variables to be set for data storage. Please follow the nnU-Net documentation on how to set `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.

## Usage

To train a model with one of the new loss functions, you can use the custom trainers with the standard nnU-Net commands.

### Available Trainers

-   `nnUNetTrainerLHLoss`
-   `nnUNetTrainerLAHLoss`
-   `nnUNetTrainerLAHsymLoss`
-   `nnUNetTrainerCEandLHLoss`
-   `nnUNetTrainerCEandLAHLoss`
-   `nnUNetTrainerCEandLAHsymLoss`

### Example Commands

**Train a model:**
```bash
nnUNetv2_train [DATASET] [CONFIGURATION] [FOLD] -tr [TRAINER_NAME]
```
For example, to train with `nnUNetTrainerLAHLoss`:
```bash
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainerLAHLoss
```
(Here, `2` is the Dataset ID for Pancreas, `3d_fullres` is the configuration, and `0` is the fold).

**Find the best configuration:**
```bash
nnUNetv2_find_best_configuration [DATASET] -c [CONFIGURATION] -tr [TRAINER_NAME]
```
For example:
```bash
nnUNetv2_find_best_configuration 2 -c 2d 3d_fullres -tr nnUNetTrainerLAHLoss
```

## Dataset and Data Splits

The experiments in the paper were performed on the Pancreas dataset from the Medical Segmentation Decathlon (Task07_Pancreas).

The data splits used for training, validation, and testing are provided in the `dataset_splits/Task07_Pancreas/` directory.

-   `train.txt`
-   `validation.txt`
-   `test.txt`

To use these splits for your training, you need to create a `splits_final.json` file in your preprocessed Pancreas dataset directory (e.g., `nnUNet_preprocessed/Dataset007_Pancreas/splits_final.json`).

The `splits_final.json` file should have the following format:
```json
[
    {
        "train": [
            "pancreas_001",
            "pancreas_005",
            ...
        ],
        "val": [
            "pancreas_040",
            "pancreas_042",
            ...
        ]
    }
]
```

**Note:** The case identifiers in the `.json` file should not include the `_0000.nii.gz` suffix. You can generate this file using a script that reads from the `train.txt` and `validation.txt` files.

## Project Structure

```
├── dataset_splits/
│   └── Task07_Pancreas/
│       ├── test.txt
│       ├── train.txt
│       └── validation.txt
├── nnunetv2/
│   ├── ... (original nnU-Net v2 files)
│   └── training/
│       ├── loss/
│       │   ├── lh.py
│       │   ├── lah.py
│       │   └── lah22.py
│       └── nnUNetTrainer/
│           └── variants/
│               └── loss/
│                   ├── nnUNetTrainerLHLoss.py
│                   ├── nnUNetTrainerLAHLoss.py
│                   └── ...
└── README.md
``` 