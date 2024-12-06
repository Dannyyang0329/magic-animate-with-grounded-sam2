#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <video_filename> <source_image_filename>"
    exit 1
fi

VIDEO_FILE="$1"
SOURCE_IMAGE_FILE="$2"

SOURCE_IMAGE_PATH="/$(pwd)/img_sample/${SOURCE_IMAGE_FILE}"
INPUT_VIDEO_PATH="/$(pwd)/video_sample/orig/${VIDEO_FILE}"
EDITED_VIDEO_PATH="/$(pwd)/video_sample/edited/edited_${VIDEO_FILE}"
DENSEPOSE_OUTPUT_PATH="/$(pwd)/densepose/output_videos/pose_${VIDEO_FILE}"

if [ ! -f "$INPUT_VIDEO_PATH" ]; then
    echo "Error: File $INPUT_VIDEO_PATH does not exist! Exiting..."
    exit 1
fi

if [ ! -f "$SOURCE_IMAGE_PATH" ]; then
    echo "Error: File $SOURCE_IMAGE_PATH does not exist! Exiting..."
    exit 1
fi


echo "Processing video edit..."
python magic-animate/video_edit.py --input "$INPUT_VIDEO_PATH" --output "$EDITED_VIDEO_PATH"
if [ $? -ne 0 ]; then
    echo "Error: video_edit.py failed for $VIDEO_FILE! Exiting..."
    exit 1
fi

echo "Processing Vid2densepose..."
cd densepose || exit 1 
python video2densepose.py -m model/model_final_162be9.pkl -i "$EDITED_VIDEO_PATH" -o "$DENSEPOSE_OUTPUT_PATH"
if [ $? -ne 0 ]; then
    echo "Error: video2densepose.py failed for $VIDEO_FILE! Exiting..."
    exit 1
fi
cd - || exit 1 

echo "Densepose Done! Densepose result: ${DENSEPOSE_OUTPUT_PATH}"

YAML_FILE="/$(pwd)/magic-animate/configs/prompts/animation_from_vid.yaml"


echo "Updating $YAML_FILE with new inputs..."

# update yaml
python3 <<EOF
import yaml

with open("$YAML_FILE", "r") as f:
    config = yaml.safe_load(f)

config["source_image"] = ["$SOURCE_IMAGE_PATH"]
config["video_path"] = ["$DENSEPOSE_OUTPUT_PATH"]

with open("$YAML_FILE", "w") as f:
    yaml.safe_dump(config, f)

print("YAML update success！")
EOF

echo "Processing Magic-animate..."
cd magic-animate || exit 1 
python3 -m magicanimate.pipelines.animation --config "$YAML_FILE"
if [ $? -eq 0 ]; then
    echo "Magic-animate processing completed successfully!"
else
    echo "Error: Magic-animate processing failed!"
    exit 1
fi
cd - || exit 1 
