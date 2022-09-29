#pip install deepface
#pip install matplotlib

# VGG-Face + Retinaface is a good one.
import time
import sys, os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from deepface import DeepFace


start_time = time.time()
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

def printHelp():
  print(f""" Usage:
    python {sys.argv[0]} <file picture> 
  """)


if len(sys.argv) < 2:
  printHelp()
  exit()
img = sys.argv[1]
if not os.path.exists(img):
  print(f"File '{img}' not found")
  exit()

df = DeepFace.find(img_path = img, db_path = "./face_db", model_name=models[0], detector_backend=backends[4])

if len(df) == 0:
  print("No match in the database")
  exit()

candidate = df['identity'][0]
header = df.columns.values[1]
similarity_consine = df[header][0]
similarity_persen = round(1 - similarity_consine, 2)* 100
#verification = DeepFace.verify(img1_path=candidate, img2_path=img, model_name=models[0], detector_backend="retinaface")
#print(f"candidate => {candidate}, match = {{verification['verified']}}   distance = {verification['distance']}")
print("="*30)
print("RESULT")
print("="*30)
matchName = candidate.split("/")[-1].split(".")[0]
print(matchName)
print("="*30)
print("--- %s seconds ---" % (time.time() - start_time))
# plot image
fig = plt.figure(figsize=(10,6))
Image1 = mpimg.imread(img)
Image2 = mpimg.imread(candidate)
fig.add_subplot(1, 2, 1)
plt.imshow(Image1)
plt.axis("off")
plt.title("Source")

fig.add_subplot(1, 2, 2)
plt.imshow(Image2)
plt.axis("off")
plt.title(f"{matchName}")

plt.suptitle(f"Match similarity = {similarity_persen} %")
plt.show()

