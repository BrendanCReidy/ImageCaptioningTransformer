import nltk
import ImageCaptioning
import os
import json
import collections
import random

random.seed(42)

#caption = ImageCaptioning.predict("/home/user/Documents/ImageCaptioningTransformer/evaluation_images/baseball.jpg", noise=0.75)
#print(caption)

captions_folder = "data/annotations"
image_folder = "data/val2017"
f = open(captions_folder + '/captions_val2017.json')
annotations = json.load(f)
f.close()

image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
  caption = f'<start> {val["caption"]} <end>'
  image_path = image_folder + "/" + '%012d.jpg' % (val['image_id'])
  image_path_to_caption[image_path].append(caption.split(" "))

image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

bleu1 = 0
bleu2 = 0
bleu3 = 0
bleu4 = 0

n_samples = 0
for imgfile in image_paths:
  prediction = ImageCaptioning.predict(imgfile, noise=1)
  print("*"*30)
  print("OUTPUT:", prediction.split(" ")[:-1])
  print("BASELINE:")
  captions_formatted = []
  for caption in image_path_to_caption[imgfile]:
    sentence = []
    for word in caption[1:-1]:
      sentence.append(word.lower())
    captions_formatted.append(sentence)
  for caption in captions_formatted:
    print("\t", caption)
  a,b,c,d = nltk.translate.bleu_score.sentence_bleu(captions_formatted, prediction.split(" ")[:-1], weights = [(1,0,0,0),(1./2., 1./2.), (1./3., 1./3., 1./3.),(1./4., 1./4., 1./4., 1./4.)])
  bleu1+=a
  bleu2+=b
  bleu3+=c
  bleu4+=d
  print("BLEU: ",a,b,c,d)
  n_samples+=1
  if n_samples>=500:
    break

print("Bleu-1 score:", bleu1 / n_samples)
print("Bleu-2 score:", bleu2 / n_samples)
print("Bleu-3 score:", bleu3 / n_samples)
print("Bleu-4 score:", bleu4 / n_samples)