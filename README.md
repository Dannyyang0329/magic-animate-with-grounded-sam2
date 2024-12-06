# MagicAnimate using Grounded-SAM-2

# Installation

* Grounded-SAM-2
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


