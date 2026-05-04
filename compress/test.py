import numpy as np
from PIL import Image
from compressimg import COMPRESSIMG  # We can use either HEVC or VVC
from pathlib import Path

CODEC_DIR = Path(__file__).parent / 'codec'

def test1():
    import matplotlib.pyplot as plt
    # Load the example image 'a.png'
    try:
        image = Image.open(CODEC_DIR / 'a.png')
    except FileNotFoundError:
        print(f"Error: Could not find 'a.png' in the '{CODEC_DIR}' directory.")
        exit()
        
    imagearray = np.array(image, dtype=np.uint8)

    # --- HEVC test ---
    print("--- Testing HEVC Codec ---")
    hevc_compressor = COMPRESSIMG(model='hevc', quality=32)
    
    # Create YUV file for inspection
    hevc_compressor.net._rgb_to_yuv(imagearray, CODEC_DIR / 'a.yuv')
    print(f"Generated '{CODEC_DIR / 'a.yuv'}' for inspection.")

    # Compress and reconstruct
    rec_hevc, bpp_hevc = hevc_compressor.net.compress(imagearray)
    print(f"HEVC BPP: {bpp_hevc:.4f}")

    # Save reconstructed image
    rec_image_hevc = Image.fromarray(rec_hevc)
    rec_image_hevc.save(CODEC_DIR / 'rec_a.png')
    print(f"Saved reconstructed HEVC image to '{CODEC_DIR / 'rec_a.png'}'")

    # --- VVC test ---
    print("\n--- Testing VVC Codec ---")
    vvc_compressor = COMPRESSIMG(model='vvc', quality=32)
    rec_vvc, bpp_vvc = vvc_compressor.net.compress(imagearray)
    print(f"VVC BPP: {bpp_vvc:.4f}")
    rec_image_vvc = Image.fromarray(rec_vvc)
    rec_image_vvc.save(CODEC_DIR / 'rec_a_vvc.png')
    print(f"Saved reconstructed VVC image to '{CODEC_DIR / 'rec_a_vvc.png'}'")

    # --- Comparison with a CompressAI model ---
    print("\n--- Testing CompressAI Model ---")
    cpr = COMPRESSIMG(model='mbt2018_mean', quality=4)
    rec_compressai = cpr.compress(imagearray)
    print(f"mbt2018_mean BPP: {cpr.bpp[0]:.4f}")

    # --- Display results ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(imagearray)
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(rec_hevc)
    axs[0, 1].set_title(f'HEVC (QP=32)\nBPP: {bpp_hevc:.4f}')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(rec_vvc)
    axs[1, 0].set_title(f'VVC (QP=32)\nBPP: {bpp_vvc:.4f}')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(rec_compressai)
    axs[1, 1].set_title(f'mbt2018_mean (Q=4)\nBPP: {cpr.bpp[0]:.4f}')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
def test2():
    image = np.array(Image.open("img_distort/example.png"))
    cpr = COMPRESSIMG(model="distort",type=23)
    rec = cpr.compress(image)
    rec_image = Image.fromarray(rec)
    rec_image.save("img_distort/rec_example.png")
    print(f"Saved reconstructed image to '../img_distort/rec_example.png'")
if __name__ == "__main__":
    test2()
