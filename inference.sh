cuda_device=1

# Grounded-SAM-2
input_video_name="input_video"
input_video_path="../data/input_videos/${input_video_name}.mp4"
sam2_output_path="../data/output_videos/${input_video_name}/video_no_bg.mp4"
sam2_src_folder="../data/output_videos/${input_video_name}/original_frames"
sam2_trc_folder="../data/output_videos/${input_video_name}/tracked_frames"
sam2_text_prompt="Human"
# DensePose
model_name="densepose_R101"
densepose_model_path="model/${model_name}.pkl"
densepose_output_path="../data/output_videos/${input_video_name}/${model_name}.mp4"
# Magic-animate
magic_animate_yaml="configs/prompts/animation_from_vid.yaml"
reference_image_path="../data/reference_images/ElonMusk.jpg"
result_video_folder="../data/output_videos/${input_video_name}/result.mp4"

source activate gsam2
if [ $? -eq 0 ]; then
    echo "Conda environment gsam2 activated successfully!"
else
    echo "Error: Failed to activate conda environment!"
    exit 1
fi

# Using Grounded-SAM-2 to get binary human mask in each frame in the reference video
echo "Get binary human mask in each frame in the reference video ..."
cd Grounded-SAM-2 || exit 1
CUDA_VISIBLE_DEVICES=$cuda_device python get_human_only_video.py \
    -i $input_video_path \
    -o $sam2_output_path \
    -p $sam2_text_prompt \
    -s $sam2_src_folder \
    -t $sam2_trc_folder
if [ $? -eq 0 ]; then
    echo "Grounded-SAM-2 processing completed successfully!"
else
    echo "Error: Grounded-SAM-2 processing failed!"
    exit 1
fi
cd - || exit 1


# Using DensePose to get densepose results in the reference video
cd Densepose || exit 1
echo "Get densepose results in the reference video ..."
echo $sam2_output_path
CUDA_VISIBLE_DEVICES=$cuda_device python video2densepose.py \
    -m $densepose_model_path \
    -i $sam2_output_path \
    -o $densepose_output_path 
if [ $? -eq 0 ]; then
    echo "Finish getting densepose results in the reference video"
else
    echo "Error: Failed to get densepose results in the reference video"
    exit 1
fi
cd - || exit 1

# exit the conda environment
conda deactivate || exit 1




# Using Magic-animate to animate the reference video
cd MagicAnimate || exit 1

conda activate m-anim
if [ $? -eq 0 ]; then
    echo "Conda environment m-anim activated successfully!"
else
    echo "Error: Failed to activate conda environment!"
    exit 1
fi

#  Update the yaml file with new inputs
echo "Update $magic_animate_yaml with new inputs ..."
# update yaml
python3 <<EOF
import yaml

with open("$magic_animate_yaml", "r") as f:
    config = yaml.safe_load(f)

config["source_image"] = ["$reference_image_path"]
config["video_path"] = ["$densepose_output_path"]

with open("$magic_animate_yaml", "w") as f:
    yaml.safe_dump(config, f)

print("YAML update success！")
EOF

echo "Processing Magic-animate..."
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m magicanimate.pipelines.animation \
    --config "$magic_animate_yaml" 

if [ $? -eq 0 ]; then
    echo "Magic-animate processing completed successfully!"
else
    echo "Error: Magic-animate processing failed!"
    exit 1
fi
cd - || exit 1 

# exit the conda environment
conda deactivate || exit 1