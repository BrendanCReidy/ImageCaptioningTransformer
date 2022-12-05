import os
import ImageCaptioning
import time
from gtts import gTTS
eval_dir = "evaluation_images"

for image_name in os.listdir(eval_dir):
    caption = ImageCaptioning.predict(os.path.join(eval_dir, image_name))
    print("[" + image_name + "] Prediction:   \"" + caption + "\"")
    language = 'en'
    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    myobj = gTTS(text=caption, lang=language, slow=False)
    
    # Saving the converted audio in a mp3 file named
    # welcome 
    myobj.save("caption_audio.mp3")
    
    # Playing the converted file
    time.sleep(1)
    os.system("caption_audio.mp3")
    time.sleep(3)