import os
import cv2
import torch
import argparse
import warnings
import numpy as np
import supervision as sv

from tqdm import tqdm
from pathlib import Path
from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer


warnings.filterwarnings("ignore")


class Video2DensePose:
    def __init__(
            self,
            model_weight_path: str,
            config_file_path: str = "detectron2/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC2M_s1x.yaml"
    ) -> None:
        # Initialize Detectron2 configuration for DensePose
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.merge_from_file(config_file_path)
        self.cfg.MODEL.WEIGHTS = model_weight_path
        self.predictor = DefaultPredictor(self.cfg)
        print("Model loaded successfully")
    
    def predict(
            self, 
            input_video_path: str,
            output_video_path: str,
        ) -> None:
        print(f"Input video path:\t {input_video_path}")
        print(f"Output video path:\t {output_video_path}")

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            with torch.no_grad():
                outputs = self.predictor(frame)["instances"]

            results = DensePoseResultExtractor()(outputs)

            # MagicAnimate uses the Viridis colormap for their training data
            cmap = cv2.COLORMAP_VIRIDIS
            # Visualizer outputs black for background, but we want the 0 value of
            # the colormap, so we initialize the array with that value
            arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
            out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)
            out.write(out_frame)

        # Release resources
        cap.release()
        out.release()

        print("Video processed successfully")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to the DensePose model weight file")
    # parser.add_argument("--config_path", "-c", type=str, required=True, help="Path to the DensePose config file")
    parser.add_argument("--input_video_path", "-i", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_video_path", "-o", type=str, required=True, help="Path to the output video file")
    args = parser.parse_args()

    video2densepose = Video2DensePose(args.model_path)#, config_file_path=args.config_path)
    video2densepose.predict(args.input_video_path, args.output_video_path)