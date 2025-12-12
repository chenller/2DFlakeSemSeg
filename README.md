## 1. Dataset

Due to the large size of the dataset, it has been uploaded to Hugging Face using split zip archives. Please follow the steps below to download and extract the data:

🔗 Dataset Homepage: [https://huggingface.co/datasets/openhcsanyu/2DFlakeSemSeg](https://huggingface.co/datasets/openhcsanyu/2DFlakeSemSeg)



Linux: Download the files to your server and run the following commands:
```shell
7z x graphene.zip.001
7z x MoS2.zip.001
```

Windows: Download the files to your local machine and extract them using a decompression tool (e.g., 7-Zip or WinRAR).

- Note: All split archive parts (e.g., .001, .002, etc.) must be placed in the same directory before extraction.
- Note: Alternatively, unzip on Windows and upload to the server.

The directory structure after decompression is as follows:
```text
/path/to/MaterialName/
├── annotations_semseg/
│   ├── train2024/        # Segmentation masks for training images (.png)
│   └── val2024/          # Segmentation masks for validation images (.png)
├── train2024/            # training images
└── val2024/              # validation images
```


## 2. Installation

**Step 1.** install mmseg: Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.


**Step 2.** Install `mmseg-extension` and `mmseg-extension-lib`

```shell
git clone https://github.com/chenller/mmseg-extension.git
cd mmseg-extension
bash install.sh
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
**Step 3.** Install `2DFlakeSemSeg`
```shell 
git clone https://github.com/chenller/2DFlakeSemSeg.git
```

**Step 4.** Install `mmseg-activelearning-extension`
```shell 
cd ./lib/mmseg-activelearning-extension
python setup.py install
cd ./lib/mmseg-2dmat
python setup.py install
```


## Usage
### Train the Model

Model training is performed by executing the train.sh script. This shell script typically encapsulates the full training pipeline, including environment setup, hyperparameter configuration, and invocation of the actual training program. Common contents of train.sh may include:
- Setting up the Python environment (e.g., activating a virtual environment or Conda environment);
- Defining paths to the training dataset, model checkpoints, and log directories;
- Specifying training hyperparameters such as learning rate, batch size, number of epochs, etc.;
- Calling the main training script (e.g., train.py) with appropriate arguments.

To start training, run:

```shell
bash ./train.sh
```

The pretrained model can be downloaded here: [flash_intern_image_b_1k_224.pth](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_b_1k_224.pth)

### Active Learning and Data Selection

The `sample_image_filepath.py` script implements the active learning workflow and data selection logic. It is designed to identify the most informative unlabeled samples for annotation under limited labeling budgets. Typical functionalities include:
- Loading a pre-trained or currently trained model checkpoint;
- Performing inference on the unlabeled pool to compute uncertainty scores (e.g., predictive entropy, least confidence, or margin sampling);
- Selecting a subset of samples based on a specified acquisition strategy (e.g., Top-K highest uncertainty);
- Outputting the selected sample IDs, file paths, or generating annotation task lists.

To run the active learning selection step, execute:

```shell
python ./sample_image_filepath.py
```

The trained weights on our dataset can be downloaded here: [pretrained_2dflake](https://huggingface.co/datasets/openhcsanyu/2DFlakeSemSeg/tree/main/pretrained_2dflake)
