cd Grounded-SAM-2

# Download the models (sam2)
cd checkpoints
bash download_ckpts.sh
if [ $? -ne 0 ]; then
    echo "Failed to download the sam2 models"
    exit 1
else
    echo "Downloaded sam2 models"
fi

# Download the models (grounding dino)
cd ../gdino_checkpoints
bash download_ckpts.sh
if [ $? -ne 0 ]; then
    echo "Failed to download the grounding dino models"
    exit 1
else
    echo "Downloaded grounding dino models"
fi

echo "Downloaded all models successfully!"
