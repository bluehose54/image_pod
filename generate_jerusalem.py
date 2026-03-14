from diffusers import DiffusionPipeline
import torch

print("Loading model...")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

prompt = "A sweeping, hyper-realistic, high-resolution aerial photograph capturing the entire walled city of ancient Jerusalem, viewed from a slightly elevated vantage point to the northeast."
print(f"Generating image for: '{prompt}'")

# Generate and save the image directly to the disk
image = pipe(prompt=prompt).images[0]
image.save("/workspace/jerusalem.png")
print("Image saved successfully to /workspace/jerusalem.png")
