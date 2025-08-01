# using_fer_detecting_the_emotions
🧠 Facial Expression Recognition using FER & OpenCV on Google Colab
This project uses the FER (Facial Expression Recognition) library along with OpenCV to detect facial emotions in images without requiring any model training. It's ideal for quick prototyping or applications like emotion-aware systems, smart apps, etc.

✅ Features
No model training required (uses pre-trained deep learning models).

Detects emotions in static images.

Uses Google Colab (no installation headaches).

Displays detected emotion and confidence with a bounding box.

🛠️ Requirements
Just install a few Python libraries inside Google Colab:

!pip install fer opencv-contrib-python
📁 Upload Image in Google Colab
Use the following to upload your image:

from google.colab import files
uploaded = files.upload()
📷 Detect Emotion from Image
Here’s the full working code:

from fer import FER
import cv2
import matplotlib.pyplot as plt

# Upload and read the image
img = cv2.imread("your_image_name.jpg")  # Replace with the uploaded file name
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize detector (MTCNN helps with better detection)
detector = FER(mtcnn=True)

# Get all detected emotions
results = detector.detect_emotions(img_rgb)
top_emotion = detector.top_emotion(img_rgb)

# Print result in console
print("Detection result:", results)
print("Top emotion:", top_emotion)

# Draw result on image
if results and top_emotion:
    emotion, score = top_emotion
    (x, y, w, h) = results[0]["box"]
    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img_rgb, f"{emotion} ({int(score*100)}%)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Show image with annotation
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Detected Emotion")
plt.show()
📊 Supported Emotions
FER detects these emotions:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

💡 Notes
FER internally uses pretrained deep learning models with good accuracy.

No manual dataset or model training is needed.

You can test multiple images by uploading them one by one.

Works well for grayscale or RGB facial images.
🔚 No training required — just run and detect!
