import distort_lib
import PIL.Image as Image
import numpy as np
imagearray = np.array(Image.open("example.png"))
distorter = distort_lib.ImageDistortion(imagearray)

custom_params = {
        'additive_gaussian_noise': {"mean": 0, "std": 25},
    }
distorter.save_imgs(params=custom_params)