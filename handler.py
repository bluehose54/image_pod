import runpod
from runpod.serverless.utils.rp_cleanup import clean
from diffusers import FluxPipeline
import torch
import base64
from io import BytesIO
import random
import traceback
import os

# --- 1. NETWORK VOLUME & AUTH SETUP ---
VOLUME_PATH = "/runpod-volume/huggingface-cache"
os.makedirs(VOLUME_PATH, exist_ok=True)

# FLUX is a gated model. This pulls the token from your RunPod Environment Variables.
HF_TOKEN = os.environ.get("HF_TOKEN")

# Performance Upgrade for modern GPUs
torch.backends.cuda.matmul.allow_tf32 = True

print("Checking Network Volume for FLUX.1 [dev] model. This may take 5-10 minutes on the first run...")

try:
    # --- 2. LOAD FLUX PIPELINE ---
    # Notice we are using FluxPipeline and bfloat16 (which FLUX requires)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir=VOLUME_PATH,
        token=HF_TOKEN
    )
    
    # CRITICAL: This prevents 24GB GPUs from crashing by offloading idle parts of the 23GB model to system RAM
    pipe.enable_model_cpu_offload()
    print("FLUX loaded successfully!")
    
except Exception as e:
    print(f"❌ Failed to load FLUX: {e}")
    print("Did you forget to add your HF_TOKEN to the RunPod Endpoint Environment Variables?")
    raise e

def handler(job):
    try:
        job_input = job.get('input', {})
        
        # --- INPUT EXTRACTION ---
        prompt = job_input.get('prompt', 'A sweeping, hyper-realistic aerial photograph of ancient Jerusalem.')
        
        width = int(job_input.get('width', 1344))
        height = int(job_input.get('height', 768))
        
        # FLUX requires fewer steps for mastery. 28-30 is the sweet spot.
        steps = int(job_input.get('steps', 28))
        
        # FLUX guidance scale is much lower than SDXL. 3.5 is standard.
        cfg_scale = float(job_input.get('guidance_scale', 3.5))
        
        print(f"Generating FLUX image ({width}x{height}) for prompt: {prompt}")
        
        seed = job_input.get('seed', random.randint(0, 1000000))
        # CPU generator is safer when using cpu_offload
        generator = torch.Generator(device="cpu").manual_seed(seed) 
        
        # --- GENERATION ---
        # Note: No negative_prompt! FLUX doesn't need it.
        image = pipe(
            prompt=prompt,
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

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error during generation: {error_msg}")
        traceback.print_exc() 
        return {
            "status": "failed",
            "error": error_msg
        }
        
    finally:
        clean()

# Start the Serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
