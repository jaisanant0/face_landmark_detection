from imutils import face_utils
import dlib
import cv2
import os
 
# facial landmark detector and  predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(-1)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #video codec
out = cv2.VideoWriter(os.getcwd()+'/output_video1.avi',fourcc,5,(int(cap.get(3)), int(cap.get(4))),True)

while True:
    # load the input image and convert it to grayscale(choice - remove # and replace image with gray)
    ret, image = cap.read()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale/image 
    rects = detector(image, 1)
    print('Number of faces detected : ' + str(len(rects)))

    #to know the scores of detecting	
    '''dets, scores, idx = detector.run(image, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))
      '''

    # loop over the face detected
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array`
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (150, 150, 150), -1)
    
    # show the output image with the face detections + facial landmarks
    out.write(image)

    cv2.namedWindow('Landmarks',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Landmarks', 900,600)
    
    cv2.imshow("Landmarks", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

