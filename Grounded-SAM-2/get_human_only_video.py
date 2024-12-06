import os
import sys
import cv2
import torch
import argparse
import warnings
import numpy as np
import supervision as sv

from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images


warnings.filterwarnings("ignore")


class GroundingSAM2():
    def __init__(
            self,
            grounding_dino_config: str="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounding_dino_checkpoint: str="gdino_checkpoints/groundingdino_swint_ogc.pth",
            sam2_checkpoint: str="checkpoints/sam2.1_hiera_large.pt",
            sam2_model_cfg: str="configs/sam2.1/sam2.1_hiera_l.yaml",
        ) -> None:
        # set the hyperparameters
        self.grounding_dino_config = grounding_dino_config
        self.grounding_dino_checkpoint = grounding_dino_checkpoint
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_cfg = sam2_model_cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.grounding_dino_config,
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device
        )
    
        # init sam image predictor and video predictor model
        self.video_predictor = build_sam2_video_predictor(self.sam2_model_cfg, sam2_checkpoint)
        self.sam2_image_model = build_sam2(self.sam2_model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model)

        self.ann_frame_idx = 0  # the frame index we interact with
        self.inference_state = None

    def get_video_frames(
        self,
        input_video_path: str,
        source_video_frame_dir: str="input_video_frames",
        ann_frame_idx: int=0
    ) -> list:
        # video_info = sv.VideoInfo.from_video_path(input_video_path)  # get video info
        frame_generator = sv.get_video_frames_generator(input_video_path, stride=1, start=0, end=None)
        # saving video to frames
        source_frames = Path(source_video_frame_dir)
        source_frames.mkdir(parents=True, exist_ok=True)
        # if the directory is not empty, remove all the files
        for file in os.listdir(source_frames):
            os.remove(os.path.join(source_frames, file))

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(source_video_frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # init video predictor state
        self.inference_state = self.video_predictor.init_state(video_path=input_video_path)
        self.ann_frame_idx = ann_frame_idx

        return frame_names

    def get_box_coordinate_using_grounding_dino(
        self,
        source_video_frame_dir: str,
        frame_names: list,
        text_prompt: str,
        box_threshold: float=0.35,
        text_threshold: float=0.25,
    ):
        # prompt grounding dino to get the box coordinates on specific frame
        img_path = os.path.join(source_video_frame_dir, frame_names[self.ann_frame_idx])
        image_source, image = load_image(img_path)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # process the box prompt for SAM2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidences = confidences.numpy().tolist()
        class_names = labels

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(image_source)
        # process the detection results
        objects = class_names

        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        return objects, masks, input_boxes


    def _register_objects_to_video_predictor(
        self,
        objects,
        input_boxes,
    ):
        # Register each object's positive points to video predictor with seperate add_new_points call
        for object_id, (label, box) in enumerate(zip(objects, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
        return out_obj_ids, out_mask_logits

    def _propagate_video_predictor(self):
        #  Propagate the video predictor to get the segmentation results for each frame
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments

    def save_densepose_video(
        self,
        objects,
        masks,
        input_boxes,
        frame_names,
        output_video_path,
        source_video_frame_dir: str,
        save_tracking_results_dir: str="tracking_results",
    ):
        out_obj_ids, out_mask_logits = self._register_objects_to_video_predictor(objects, input_boxes)
        video_segments = self._propagate_video_predictor()
        
        # Visualize the segment results across the video and save them
        if not os.path.exists(save_tracking_results_dir):
            os.makedirs(save_tracking_results_dir)
        # if the directory is not empty, remove all the files
        for file in os.listdir(save_tracking_results_dir):
            os.remove(os.path.join(save_tracking_results_dir, file))

        id_to_objects = {i: obj for i, obj in enumerate(objects, start=1)}

        for frame_idx, segments in video_segments.items():
            img = cv2.imread(os.path.join(source_video_frame_dir, frame_names[frame_idx]))
            
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            
            black_img = np.zeros_like(img)
            combined_mask = np.any(masks, axis=0).astype(np.uint8)
            combined_mask = np.expand_dims(combined_mask, axis=-1)
            anootated_frame = img * combined_mask + black_img * (1 - combined_mask)
            cv2.imwrite(os.path.join(save_tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"), anootated_frame)

            # detections = sv.Detections(
            #     xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            #     mask=masks, # (n, h, w)
            #     class_id=np.array(object_ids, dtype=np.int32),
            # )
            # # box_annotator = sv.BoxAnnotator()
            # # annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            # # label_annotator = sv.LabelAnnotator()
            # # annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[id_to_objects[i] for i in object_ids])
            # # using background black color
            # mask_annotator = sv.MaskAnnotator()
            # black_img = np.zeros_like(img)
            # annotated_frame = mask_annotator.annotate(scene=black_img, detections=detections)
            # # annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
            # # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            # cv2.imwrite(os.path.join(save_tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
                

        # Convert the annotated frames to video
        create_video_from_images(save_tracking_results_dir, output_video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounding SAM2 on custom video input")
    parser.add_argument("--input_video_path", "-i", type=str, help="Path to the input video")
    parser.add_argument("--output_video_path", "-o", type=str, help="Path to the output video")
    parser.add_argument("--text_prompt", "-p", type=str, help="Text prompt for grounding DINO")
    parser.add_argument("--ann_frame_idx", "-a", type=int, default=0, help="Frame index to annotate")
    parser.add_argument("--source_video_frame_dir", "-s", type=str, default="../intermediate_data/input_video_frames", help="Directory to save the video frames")
    parser.add_argument("--save_tracking_results_dir", "-t", type=str, default="../intermediate_data/tracking_results", help="Directory to save the tracking results")
    args = parser.parse_args()

    grounding_sam2 = GroundingSAM2()
    frame_names = grounding_sam2.get_video_frames(
        args.input_video_path,
        args.source_video_frame_dir,
        args.ann_frame_idx
    )
    objects, masks, input_boxes = grounding_sam2.get_box_coordinate_using_grounding_dino(
        args.source_video_frame_dir,
        frame_names,
        args.text_prompt
    )
    grounding_sam2.save_densepose_video(
        objects,
        masks,
        input_boxes,
        frame_names,
        args.output_video_path,
        args.source_video_frame_dir,
        args.save_tracking_results_dir
    )