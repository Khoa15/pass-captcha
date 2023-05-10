from torch import Tensor, load, device
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

class Captcha():
    def __init__(self, img_path, model):
        self.img_path = img_path
        self.model = model
    def __save__(self, path):
        pass
    def load(self, path):
        try:
            img = Image.open(path)
            return img
        except:
            return "Error path"
    def transform(self, img):
        try:
            return transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ])(img)
        except:
            return "Error transform"

    def __predict__(self):
        try:
            img = self.load(self.img_path)
            img = img.convert("L")
            img = self.transform(img).unsqueeze(0)
            #img = Tensor(img).cpu()
            pred = self.model(img)

            c0 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[0:ALL_CHAR_SET_LEN])]
            c1 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN:ALL_CHAR_SET_LEN*2])]
            c2 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*2:ALL_CHAR_SET_LEN*3])]
            c3 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*3:ALL_CHAR_SET_LEN*4])]
            c = '%s%s%s%s' % (c0, c1, c2, c3)
            return c
        except:
            return "Error predict"

if __name__ == "__main__":
    model = load("own_captcha_resnet18_mix_data.pth", map_location=device('cpu'))
    model = model.eval()
    cap = Captcha(img_path="./captcha_datasets/0A14.png", model=model)
    print(cap.__predict__())