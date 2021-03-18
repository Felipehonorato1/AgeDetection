import cv2
import os
# cap = cv2.VideoCapture()


class img_processing():
    def __init__(self, img_path, output_path):
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.img_path = img_path
        self.output_path = output_path
        self.img_id = 0

    def resize(self, img):
        return cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)

    def crop_images(self):
        for img in os.listdir(self.img_path):
            img_read = cv2.imread(f'{self.img_path}/{img}')

            gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                if (w > 200) & (h > 200):
                    img_crop = img_read[y:y+h, x:x+w]
                    img_resized = self.resize(img_crop)
                    self.img_id += 1
                    cv2.imwrite(f'{self.output_path}/{img}', img_resized)

        print(
            f'DONE; {self.img_id} images cropped out of {len(os.listdir(self.img_path))}')
