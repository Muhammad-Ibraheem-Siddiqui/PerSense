from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import torch
from PIL import Image , ImageChops
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# # load support image and mask
supp_mask = Image.open("/l/users/muhammad.siddiqui/Personalize-SAM/Example_images/mask.png").convert("RGB")
supp_image = Image.open("/l/users/muhammad.siddiqui/Personalize-SAM/Example_images/image.png").convert("RGB")
raw_image = ImageChops.multiply(supp_image, supp_mask)
raw_image.save("/l/users/muhammad.siddiqui/Personalize-SAM/Example_images/masked_img.png", "PNG")
# image_file = "/l/users/muhammad.siddiqui/Personalize-SAM/Example_images/masked_img.png"
image_file = "/l/users/muhammad.siddiqui/Personalize-SAM/data/Images/colorful_sneaker/01.jpg"

model_path = "mucai/vip-llava-7b"
prompt = "What is the name of the object in this image?"

args = type('Args', (), {
    "model_path": model_path,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "image_file": image_file,
    "conv_mode": None, "model_base": None, "temperature": 0.2, "top_p": None, "num_beams": 1, "max_new_tokens": 512, "sep": ",",
})()

output = eval_model(args)
print(output)
words = output.split()
last_word = words[-1]
last_word = [last_word.replace(".", "")]
print (last_word)


#load query image
image_file2 = "/l/users/muhammad.siddiqui/Personalize-SAM/Example_images/sneakers_many.png"
prompt2 = "Count the number of" + last_word[0] + "in this image?"

args = type('Args', (), {
    "model_path": model_path,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt2,
    "image_file": image_file2,
    "masked_img" : raw_image,
    "conv_mode": None, "model_base": None, "temperature": 0.2, "top_p": None, "num_beams": 1, "max_new_tokens": 512, "sep": ",",
})()

output2 = eval_model(args)
print(output2)


