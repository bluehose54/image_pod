import runpod
from runpod.serverless.utils.rp_cleanup import clean
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
import random

# 1. Load the model outside the handler for cold-start optimization
print("Loading SDXL model into VRAM...")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")
print("Model loaded successfully!")

# 2. Define the handler function
def handler(job):
    try:
        job_input = job['input']
        
        # --- ADVANCED QUALITY SETTINGS ---
        
        # The main prompt
        prompt = job_input.get('prompt', 'A sweeping, hyper-realistic aerial photograph of ancient Jerusalem.')
        
        # Negative Prompt: This is the secret to photorealism. We explicitly tell the AI what styles to avoid.
        default_negative = "cartoon, painting, illustration, worst quality, low quality, blurry, grainy, deformed, distorted, unnatural lighting, text, watermark"
        negative_prompt = job_input.get('negative_prompt', default_negative)
        
        # SDXL Widescreen WXGA Resolution (1344x768 is natively supported and highly optimized)
        width = job_input.get('width', 1344)
        height = job_input.get('height', 768)
        
        # Inference Steps: 40 is the sweet spot for SDXL base to resolve fine details (stone textures, tiny people)
        steps = job_input.get('steps', 40)
        
        # Guidance Scale (CFG): 7.5 is the optimal balance between following your prompt and maintaining natural realism
        cfg_scale = job_input.get('guidance_scale', 7.5)
        
        print(f"Generating high-res image for prompt: {prompt}")
        
        # Use a random seed, or allow the user to pass a specific one for reproducible results
        seed = job_input.get('seed', random.randint(0, 1000000))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # 3. Generate the image with all parameters applied
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]

        # 4. Convert to Base64 (Using PNG instead of JPEG for lossless compression to maintain maximum sharpness)
        buffered = BytesIO()
        image.save(buffered, format="PNG") 
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "prompt_used": prompt,
            "seed_used": seed,
            "image_base64": img_str
        }
    finally:
        # Always clean up the temporary VRAM/RAM buffers to prevent memory leaks
        clean()

# 5. Start the Serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
