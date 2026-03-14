import runpod
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO

# 1. Load the model outside the handler to optimize cold starts [2, 11]
print("Loading model into VRAM...")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")
print("Model loaded successfully!")

# 2. Define the handler function [7]
def handler(job):
    job_input = job['input']

    # Fallback prompt [7]
    prompt = job_input.get('prompt', 'A sweeping, hyper-realistic aerial photograph of ancient Jerusalem.')

    print(f"Generating image for prompt: {prompt}")

    # Generate the image
    image = pipe(prompt=prompt).images[0]

    # Convert the PIL image to a Base64 string for the JSON response [7]
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "image_base64": img_str
    }

# 3. Start the serverless worker (Move this to the top level) [7, 16]
runpod.serverless.start({"handler": handler})
