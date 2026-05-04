import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

MODEL_DIR = "/mnt/shared-storage-user/zhangjianbo/models/playground-v25-fp16"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="robotics scene")
    parser.add_argument("--negative_prompt", default="blurry, distorted, artifacts, extra objects, wrong structure")
    parser.add_argument("--strength", type=float, default=0.28)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    ).to(device)

    image = Image.open(args.input).convert("RGB").resize((1024, 1024))

    generator = torch.Generator(device=device).manual_seed(args.seed)

    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=image,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
    ).images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    print(f"saved: {output_path}")

if __name__ == "__main__":
    main()
