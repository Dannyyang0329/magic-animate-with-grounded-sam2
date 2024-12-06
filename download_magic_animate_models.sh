cd MagicAnimate/pretrained_models
if [ ! -d "MagicAnimate" ]; then
    echo "Downloading MagicAnimate model..."
    git lfs clone https://huggingface.co/zcxu-eric/MagicAnimate
else
    echo "MagicAnimate model already exists"
fi

if [ ! -d "sd-vae-ft-mse" ]; then
    echo "Downloading sd-vae-ft-mse model..."
    git lfs clone https://huggingface.co/stabilityai/sd-vae-ft-mse
else
    echo "sd-vae-ft-mse model already exists"
fi

# get the HUGGINGFACE_TOKEN from .env file
if [ ! -d "stable-diffusion-v1-5" ]; then
    echo "Downloading stable-diffusion-v1-5 model..."
    git lfs clone "https://huggingface.co/benjamin-paine/stable-diffusion-v1-5"
else
    echo "stable-diffusion-v1-5 model already exists"
fi