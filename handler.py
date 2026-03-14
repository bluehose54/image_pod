import runpod
from runpod.serverless.utils.rp_cleanup import clean
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import base64
from io import BytesIO
import random
import traceback

# --- 1. PERFORMANCE UPGRADE ---
# Enable TensorFloat-32 (TF32) for 20-30% faster generation on Ampere+ GPUs (RTX 30xx/40xx/A-series)
torch.backends.cuda.matmul.allow_tf32 = True

print("Loading SDXL model into VRAM...")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# --- 2. SCHEDULER UPGRADE ---
# Swap to DPM++ 2M Karras for maximum photorealism and crisp micro-details
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, 
    use_karras_sigmas=True
)

pipe.to("cuda")
print("Model loaded successfully!")

def handler(job):
    try:
        job_input = job.get('input', {})
        
        # --- INPUT EXTRACTION ---
        prompt = job_input.get('prompt', 'A sweeping, hyper-realistic aerial photograph of ancient Jerusalem.')
        default_negative = "cartoon, painting, illustration, worst quality, low quality, blurry, grainy, deformed, distorted, unnatural lighting, text, watermark"
        negative_prompt = job_input.get('negative_prompt', default_negative)
        
        width = int(job_input.get('width', 1344))
        height = int(job_input.get('height', 768))
        steps = int(job_input.get('steps', 40))
        cfg_scale = float(job_input.get('guidance_scale', 7.5))
        
        print(f"Generating image ({width}x{height}) for prompt: {prompt}")
        
        seed = job_input.get('seed', random.randint(0, 1000000))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # --- GENERATION ---
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]

        # --- ENCODING ---
        buffered = BytesIO()
        image.save(buffered, format="PNG") 
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "prompt_used": prompt,
            "seed_used": seed,
            "image_base64": img_str
        }

    # --- 3. ERROR HANDLING UPGRADE ---
    except Exception as e:
        # If the code fails, safely catch the error and return it to the client as JSON
        error_msg = str(e)
        print(f"❌ Error during generation: {error_msg}")
        traceback.print_exc() # Prints the exact line of the failure to your RunPod logs
        
        return {
            "status": "failed",
            "error": error_msg
        }
        
    finally:
        # Always clean up the temporary VRAM/RAM buffers to prevent memory leaks
        clean()

# Start the Serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
