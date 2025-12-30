import torch
from diffusers import ZImagePipeline

out="/workspace/Ovi/inputs/zimage_test.png"

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
)

# VRAM sparen (macht es langsamer, aber hilft gegen OOM)
pipe.enable_model_cpu_offload()

prompt = "Close-up ASMR food scene. A cute brown monkey sits at a small wooden table in a cozy kitchen, facing the camera. On a plate is a crispy schnitzel with lemon wedge and a small side salad. Soft warm lighting, shallow depth of field, ultra detailed food texture, stable camera, no text, no watermark."

img = pipe(
    prompt=prompt,
    height=768,
    width=768,
    num_inference_steps=9,  # ergibt 8 DiT forwards
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

img.save(out)
print("saved ->", out)
