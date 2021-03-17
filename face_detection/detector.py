import cv2
# cap = cv2.VideoCapture()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# while True:
#     _, img = cap.read()
img = cv2.imread('ppl2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

i = 0
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    img_crop = img[y:y+h, x:x+w]
    cv2.imwrite(f'crop{i}.jpg', img_crop)
    i += 1
