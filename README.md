# Causality for Machine Learning

This repo accompanies the prototype code from our [report](https://ff13.fastforwardlabs.com/), specifically the Invariant Risk Minimization (IRM) approach discussed in chapters 3 & 4.

### Setup environment

```
conda create --name irm_env python=3.7 ipykernel
conda activate irm_env
conda install pip
pip install -r requirements.txt
```

### Setup data

Steps to generate the WildCam dataset used for this experiment. 

- **Step 1** : Download the Camera Traps (or Wild Cams) dataset - [iWildCam 2019](https://github.com/visipedia/iwildcam_comp) using Kaggle.
    - Use Kaggle account to download data. This involves first creating a new Kaggle API from your account and then downloading the `kaggle.json` file in `[user-home]/.kaggle` folder. If there is no `.kaggle` folder yet, create it and then move the `kaggle.json` to the `.kaggle` folder.
    - Install kaggle package, `conda install -c conda-forge kaggle`
    - `chmod 600 ~/.kaggle/kaggle.json`
    - Download data, `kaggle competitions download -c iwildcam-2019-fgvc6`. This is a 44GB file!
    - `unzip iwildcam-2019-fgvc6.zip -d ./iWildCam`
    - `unzip train_images.zip -qd ./train`. NOTE: The test set images are unlabeled so are being ignored from our experiments. Instead, we create a test set from the training data in the next step.

- **Step 2** :  
    - Run `python create_denoised_data.py` - This creates a new directory `./data/wildcam_denoised` consisting of the images we used for training and testing both the IRM and ERM models. The list of images are available in `./data/train_test_filenames.json`.

### Repo structure

```
.
├── ./create_denoised_data.py
├── ./data
│   ├── ./data/train_test_filenames.json
│   └── ./data/wildcam_denoised
│       ├── ./data/wildcam_denoised/test
│       ├── ./data/wildcam_denoised/train_43
│       └── ./data/wildcam_denoised/train_46
├── ./dataset.py
├── ./ERM_results.out
├── ./IRM_results.out
├── ./main.py
├── ./models
│   └── ./models/wildcam_denoised_121_0.001_0_0.0_ERM.pth
│   └── ./models/wildcam_denoised_121_0.001_40_10000.0_IRM.pth
├── ./models.py
├── ./README.md
├── ./requirements.txt
└── ./train.py

```

### Training

- Simply run `python main.py` to train an ERM/ IRM model. The arguments `penalty_anneal_iters` and `penalty_weight` when set to 0 trains an ERM model.
- The model is saved in the `./models` folder

### Results

- Results from training both IRM and ERM are available in `ERM_results.out` and `IRM_results.out` files

### Interpretability

- The `model_explanation_irm.ipynb` notebook provides LIME explanations for a sample image based on the IRM model. For a deeper dive into explanations and comparison between all images based on both the ERM and IRM models look at our prototype - [Scene](https://scene.fastforwardlabs.com/)

### References

Leveraged source code from the [paper](https://arxiv.org/abs/1907.02893v1):
```
@article{InvariantRiskMinimization,
    title={Invariant Risk Minimization},
    author={Arjovsky, Martin and Bottou, L{\'e}on and Gulrajani, Ishaan and Lopez-Paz, David},
    journal={arXiv},
    year={2019}
}
```