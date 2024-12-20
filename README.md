# MagicAnimate using Grounded-SAM-2

## Introduction

This project explores the integration of `Grounded-SAM-2` with `MagicAnimate` to enhance the realism of human animations. By utilizing `Grounded-SAM-2`, we aim to minimize the influence of shadows in the input reference video, which can subtly affect the accuracy of densepose generation. This refined approach seeks to improve the overall quality and realism of the videos generated by `MagicAnimate`.

## Demo

Top row: Original MagicAnimate output

Bottom row: Grounded-SAM-2 + MagicAnimate output

![Demo](demo/demo.gif)


## Installation

### Grounded-SAM-2
* Download models
    ```bash
    bash download_grounded_sam2_models.sh
    ```
* Create the environment

    ```bash
    conda create -n gsam2 python=3.10
    conda activate gsam2

    conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install gcc=9.4.0 gxx=9.4.0 -c conda-forge
    ```

* Install the dependencies

    * sam2 dependencies
        ```bash
        cd Grounded-SAM-2
        pip install -e .
        ```
    
    * grounding_dino dependencies
        ```bash
        cd Grounded-SAM-2
        pip install --no-build-isolation -e grounding_dino --use-pep517
        ```


### Densepose
* Downlaod model
    ```bash
    wget -O "Densepose/model/densepose_R101.pkl" "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2M_s1x/216245790/model_final_de6e7a.pkl"
    ```

* Install the dependencies
    ```bash
    pip install git+https://github.com/facebookresearch/detectron2@main
    ```


### Other dependencies for Grounded-SAM-2 and Densepose
    ```bash
    pip install opencv-python supervision transformers addict yapf pycocotools timm decord ninja av
    ```

### MagicAnimate

* Download models
    You need to use git-lfs to download the models. If you don't have git-lfs installed, you can install it using conda.

    NOTICE: Huggingface access token is required to download the model. You can create a token from your huggingface account

    ```bash
    conda install -c conda-forge git-lfs
    git lfs install
    ```
    ```bash
    bash download_magic_animate_models.sh
    ```

* Create a new environment

    ```bash
    conda create -n m-anim python=3.10
    conda activate m-anim
    ```

    ```
    pip install -r MagicAnimate/requirements.txt
    ```


### Usage

* Set the reference image and video paths in `inference.sh` file
* Run the following command to generate human animation video
    ```bash
    bash inference.sh
    ```