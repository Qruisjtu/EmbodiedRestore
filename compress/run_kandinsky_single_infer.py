import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Img2ImgPipeline

PRIOR_DIR = "/mnt/shared-storage-user/zhangjianbo/models/kandinsky-2-2-prior"
DECODER_DIR = "/mnt/shared-storage-user/zhangjianbo/models/kandinsky-2-2-decoder"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="robotics scene")
    parser.add_argument("--negative_prompt", default="blurry, distorted, artifacts, extra objects, wrong structure")
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
        PRIOR_DIR,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    img2img_pipe = KandinskyV22Img2ImgPipeline.from_pretrained(
        DECODER_DIR,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    image = Image.open(args.input).convert("RGB").resize((768, 768))

    prior_out = prior_pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )

    out = img2img_pipe(
        image=image,
        image_embeds=prior_out.image_embeds,
        negative_image_embeds=prior_out.negative_image_embeds,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        height=768,
        width=768,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    print(f"saved: {output_path}")

if __name__ == "__main__":
    main()
