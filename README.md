# ImageCaptioningTransformer
Repository for NLP Class Final Project

## Getting Started

step 1) download ms coco [dataset](https://cocodataset.org/#download)

step 2) unzip annotations_trainval2017.zip and train2017.zip and place into folder called "data" (at same location as the training files)

step 3) set MS_COCO_PATH inside trainTransformer.py to the directory containing ms coco

step 4) uncomment parts for trainTransformer.py where specified:

 the code will have # UNCOMMENT FOR FIRST RUN (GEN_IMAGES) and # UNCOMMENT FOR FIRST RUN (GEN_TOKENIZER)
 
 these lines of code only need to be ran once since they are generating the tokenizer and the embedded images
 
 you can leave them commented after the first run of the program
 
step 5) Train the model using python3 trainTransformer.py
 
## Evaluating Images
To evaluate an image simply add it to the Code/evaluation_images directory, then run the "evaluate_dir.py" program or the "evaluate_dir_withTTS.py" to use text to speech. Make sure there is a checkpoints folder to load the model from (this will be generated during training). Unfortunately, the models are too large to upload to github, so training must be completed from scratch in order to use the models.
