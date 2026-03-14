import runpod
from runpod.serverless.utils.rp_cleanup import clean
from diffusers import FluxPipeline
import torch
import base64
from io import BytesIO
import random
import traceback
import os

# --- 1. SYSTEM SETUP ---
VOLUME_PATH = "/runpod-volume/huggingface-cache"
os.makedirs(VOLUME_PATH, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
torch.backends.cuda.matmul.allow_tf32 = True

# We create an empty global variable for the model, but WE DO NOT LOAD IT YET.
pipe = None

def load_model():
    """Lazy loads the model only when the first job arrives."""
    global pipe
    if pipe is None:
        print("Checking Network Volume for FLUX.1 [dev] model...")
        print("If this is the first run, it will download 23GB. Do not close your terminal...")
        
        try:
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                cache_dir=VOLUME_PATH,
                token=HF_TOKEN
            )
            pipe.enable_model_cpu_offload()
            print("FLUX loaded successfully into VRAM!")
        except Exception as e:
            print(f"❌ Failed to load FLUX: {e}")
            raise e

def handler(job):
    try:
        # --- 2. THE TRIGGER ---
        # Before processing the prompt, check if the model is downloaded/loaded.
        load_model()
        
        job_input = job.get('input', {})
        
        prompt = job_input.get('prompt', 'A sweeping, hyper-realistic aerial photograph of ancient Jerusalem.')
        width = int(job_input.get('width', 1344))
        height = int(job_input.get('height', 768))
        steps = int(job_input.get('steps', 28))
        cfg_scale = float(job_input.get('guidance_scale', 3.5))
        
        print(f"Generating FLUX image ({width}x{height}) for prompt: {prompt}")
        
        seed = job_input.get('seed', random.randint(0, 1000000))
        generator = torch.Generator(device="cpu").manual_seed(seed) 
        
        # --- 3. GENERATION ---
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]

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

# --- 4. INSTANT START ---
# The worker will immediately reach this line on boot and pass RunPod's health checks!
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
