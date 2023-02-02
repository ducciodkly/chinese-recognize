import numpy as np
import cv2
from tensorflow.keras.models import load_model
model = load_model('./model_10.h5')

img = cv2.imread("../input/data-china-10/data_china_10/test/4/2535.bmp" )

image= img.astype('float32') / 255
image = np.array(image.reshape(1, 224,224,3))
pres = model.predict(image)
accuracy = model.predict_proba(image)[0]

print(np.argmax(pres))
class_labels = np.argmax(model.predict(image)[0])

lis = ['依','磨','嚏','松','冠','寺','屑','产','下','碌']
print('Kết quả=',lis[class_labels])
print('Acc=',round(max(accuracy)*100,2))