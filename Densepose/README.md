# Video to DensePose

## Requirements
You can create a new environment by `environment.yml` file with the following command:
```bash
conda env create -f environment.yml
```

Or you can install the required packages manually:
```bash
pip install numpy==1.26.4
pip install torch==2.1.0 torchvision==0.16.0 opencv-python
pip install ninja av scipy
pip install git+https://github.com/facebookresearch/detectron2@main
```

NOTICE: gcc & g++ version should be 5.4.0 or higher


## Usage
```bash
python video2densepose.py -m <model_path> -i <input_video_path> -o <output_video_path>
```

The process will take a while. For the sample video, it takes about 15 minutes.