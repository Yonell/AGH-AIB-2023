import numpy as np


class ModelDoodlesFiles:
    def __init__(self, image_size=128):
        print("Loading np bitmaps...")
        self.model_doodles_categories = open("./categories.txt", "r").read(-1).split("\n")
        self.model_doodles_excluded_categories = open("./categories.txt", "r").read(-1).split("\n")
        for i in self.model_doodles_categories:
            self.model_doodles_excluded_categories.remove(i)
        self.model_doodles_samples = []
        for i in self.model_doodles_categories:
            print("Loading " + i + "...")
            self.model_doodles_samples.append(
                np.load("./npbitmaps" + str(image_size) + "x" + str(image_size) + "/full_numpy_bitmap_" + i + ".npy"))

            self.model_doodles_categories = self.model_doodles_categories[:10000]
        for i in self.model_doodles_excluded_categories:
            print("Loading " + i + "...")
            self.model_doodles_samples.append(
                np.load("./npbitmaps" + str(image_size) + "x" + str(image_size) + "/full_numpy_bitmap_" + i + ".npy"))
            self.model_doodles_categories = self.model_doodles_categories[:10000]
