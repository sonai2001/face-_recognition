import cv2
vs = cv2.VideoCapture(0)

while True:
      _, img = vs.read() #returns ret and the frame
      cv2.imshow('Videostream',img)
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
          break
vs.release()
cv2.destroyAllWindows()
