# /workspace/Ovi/tools/zimage.py
import argparse
import json
import os

import torch
from diffusers import ZImagePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="JSON string with generation params")
    args = ap.parse_args()

    params = json.loads(args.json)

    os.environ.setdefault("HF_HOME", "/workspace/.cache/hf")

    prompt = params["prompt"]
    negative_prompt = params.get("negative_prompt") or None

    out_path = params["out_path"]
    width = int(params.get("width", 768))
    height = int(params.get("height", 768))
    steps = int(params.get("steps", 9))
    seed = int(params.get("seed", 42))
    guidance_scale = float(params.get("guidance_scale", 0.0))
    cpu_offload = bool(params.get("cpu_offload", True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    if device == "cuda" and cpu_offload:
        pipe.enable_model_cpu_offload()

    generator = torch.Generator("cuda").manual_seed(seed) if device == "cuda" else None

    img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)

    print(
        json.dumps(
            {
                "ok": True,
                "out_path": out_path,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
