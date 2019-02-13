# Face_landmark_detection
It detects the 68 important key points of the faces present in the webcam view using dlib, opencv and python.

-Detecting facial landmarks is a two step process:
   1. Localize the face in the image.
   2. Detect the key facial structures on the face ROI.

-The methods essentially try to localize and label the following facial regions:

   1. Mouth
   2. Right eyebrow
   3. Left eyebrow
   4. Right eye
   5. Left eye
   6. Nose
   7. Jaw

## Getting started 
Clone the repository and run the face_landmark_live.

We are using the model already trained, we will need to download the file shape_predictor_68_face_landmarks.dat that you can find in the repository.

## Usage 
```
python3 face_landmark_live.py 
```

NOTE : It save the output in the current directory as output_video.avi

