# ImageCaptioningTransformer
Repository for NLP Class Final Project

## Getting Started

step 1) download ms coco (dataset)[https://cocodataset.org/#download]

step 2) unzip annotations_trainval2017.zip and train2017.zip

step 3) set MS_COCO_PATH inside trainTransformer.py to the directory containing ms coco

step 4) uncomment parts for trainTransformer.py where specified:

 the code will have # UNCOMMENT FOR FIRST RUN (GEN_IMAGES) and # UNCOMMENT FOR FIRST RUN (GEN_TOKENIZER)
 
 these lines of code only need to be ran once since they are generating the tokenizer and the embedded images
 
 you can leave them commented after the first run of the program
 
