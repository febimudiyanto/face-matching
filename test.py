#pip install deepface
#pip install matplotlib


import sys

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from deepface import DeepFace

fig = plt.figure(figsize=(10,7))

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]

img = sys.argv[1]
df = DeepFace.find(img_path = img, db_path = "./face_db", model_name=models[0], detector_backend=backends[4])


candidate = df['identity'][0]
header = df.columns.values[1]
similarity_consine = df[header][0]

similarity_persen = round(1 - similarity_consine, 2)* 100
#verification = DeepFace.verify(img1_path=candidate, img2_path=img, model_name=models[0], detector_backend="retinaface")
#print(f"candidate => {candidate}, match = {{verification['verified']}}   distance = {verification['distance']}")


Image1 = mpimg.imread(img)
Image2 = mpimg.imread(candidate)
fig.add_subplot(1, 2, 1)
plt.imshow(Image1)
plt.axis("off")
plt.title("Source")

fig.add_subplot(1, 2, 2)
plt.imshow(Image2)
plt.axis("off")
plt.title(f"file =  {candidate}")

plt.suptitle(f"Match distance = {similarity_persen} %")
plt.show()

