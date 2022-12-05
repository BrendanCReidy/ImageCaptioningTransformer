import os
import ImageCaptioning
eval_dir = "evaluation_images"

for image_name in os.listdir(eval_dir):
    caption = ImageCaptioning.predict(os.path.join(eval_dir, image_name))
    print("[" + image_name + "] Prediction:   \"" + caption + "\"")