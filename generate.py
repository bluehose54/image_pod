from diffusers import DiffusionPipeline
import torch

print("Loading model...")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

prompt = "A sweeping, hyper-realistic, high-resolution aerial photograph capturing the entire walled city of ancient Jerusalem, viewed from a slightly elevated vantage point to the northeast. The shot is taken during the golden hour, with the sun low on the western horizon, casting long, dramatic shadows that emphasize the texture of the hills and the detailed stonework. In the foreground and center, the massive, newly completed outer walls and defensive towers, constructed of sun-baked Judean limestone, curve around the jagged topography, showcasing their imposing scale. The main city gates are visible and operational. Inside the walls, thousands of small, densely packed stone houses with flat roofs are clustered tightly together, broken only by narrow, winding unpaved streets bustling with tiny human figures, livestock, and market activity. The defining landmark, the First Temple (Solomon’s Temple) complex, stands prominently on Mount Moriah in the eastern half, its golden accents and white stone reflecting the warm sunlight. Other significant structures like the Palace complex and the upper city are distinguishable. Outside the city walls, terraced agricultural fields, vineyards, olive groves, and winding ancient footpaths lead toward small surrounding villages and dry, scrubby hills. The entire scene is set against the dramatic, rugged landscape of the Judaean Mountains stretching to the distant horizon under a clear, pale blue sky with a few wispy, warm-toned clouds. The composition captures the city as a cohesive fortress nestled within the natural landscape, with a sense of immense scale, historical context, and the raw energy of a living, breathing ancient capital. Grain and cinematic lighting give it an authentic film quality."
print(f"Generating image for: '{prompt}'")

# Generate and save the image
image = pipe(prompt=prompt).images[0]
image.save("/workspace/jerusalem.png")
print("Image saved successfully to /workspace/jerusalem.png")
