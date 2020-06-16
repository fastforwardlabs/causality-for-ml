# Causality for Machine Learning

This repo accompanies the prototype code discussed in our [report](https://ff13.fastforwardlabs.com/), specifically the Invariant Risk Minimization (IRM) approach discussed in chapters 3 & 4.

## Contents

Todo:
- create denoised dataset, this may require us to create a JSON file listing filenames that were actually used. Reuse some old code for this - done
- fix requirements + make sure that works on CML
- simply copy - main.py, train.py, models.py, dataset.py + changes 
- think about model explanation - a CML code / notebook demonstrating an example should be good enough? - could be simply copy-paste in CML
- simple score/ predict with CML
- Also, to take a step further, I would like to build a simple UI that allows a user to upload an image, predict IRM score and maybe also show explanations

### Data

#### Step 1: 
Use Kaggle account to download data. The Camera Traps (or Wild Cams) dataset - [iWildCam 2019](https://github.com/visipedia/iwildcam_comp).

```
conda install -c conda-forge kaggle
cd causality-for-ml/data

chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c iwildcam-2019-fgvc6
unzip iwildcam-2019-fgvc6.zip -d ./iWildCam

unzip train_images.zip -d ./train
```
NOTE: The test set images are unlabeled so are being ignored from our experiments. Instead, we create a test set from the training data in the next step.

#### Step 2: 
Run 
```
python create_denoised_data.py
```
This creates a new directory './data/wildcam_denoised' consisting of the images we used for training and testing both the IRM and ERM models.

## Structure of repo

```
.
├── ./create_denoised_data.py
├── ./data
│   ├── ./data/train_test_filenames.json
│   └── ./data/wildcam_denoised
│       ├── ./data/wildcam_denoised/processed
│       ├── ./data/wildcam_denoised/test
│       ├── ./data/wildcam_denoised/train_43
│       └── ./data/wildcam_denoised/train_46
├── ./dataset.py
├── ./ERM_results.out
├── ./figures
│   └── ./figures/wildcam_denoised_11_0.001_0_0.0_ERM.png
├── ./IRM_results.out
├── ./main.py
├── ./models
│   └── ./models/wildcam_denoised_11_0.001_0_0.0_ERM.pth
├── ./models.py
├── ./README.md
├── ./requirements.txt
└── ./train.py

```
## Results

Overview of results and model explanations


