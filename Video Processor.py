import numpy as np
from PIL import Image
import time
from tensorflow.keras.models import load_model
from IPCameraStream import IPCameraStream

model_pb = r"C:\Users\Bogingo\PycharmProjects\imageClassification\MyCNN"
model = load_model(model_pb)
camAddress = "rtsp://admin:None2021@192.168.1.64:554/Streaming/channels/1/"
images = IPCameraStream(src=camAddress)
images.start() #Start the streaming thread
class_names = ['Carrier_With_Products', 'Feeds', 'Raw Materials', 'Wheat Flour Brown',
               "Wheat Flour White"]

print("starting image processing")


def videoprocessor():
    while True:
        frame = images.read()
        Captured_frame = frame

        im = Image.fromarray(Captured_frame, 'RGB')
        im = im.resize((300, 300))
        img_array = np.array(im)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print("Predicting.... ")
        startTime = time.time()
        predictions = model.predict(img_array)
        prediction_result = np.argmax(predictions[0])
        prediction_Confidence = predictions[0][prediction_result]
        endTime = time.time()
        print("prediction time: ", endTime - startTime)
        if prediction_Confidence > 0.6:
            print("Predicted Class:", class_names[prediction_result])
            print("Prediction Confidence", prediction_Confidence)
            # print("All Classes:")

        else:
            print("Item does not belong to any of the classes")

        # TODO Update the label accordingly
        # TODO: Update the counts on the frame
        label = "{}: {:.2f}%".format(class_names[prediction_result], prediction_Confidence * 100)
        images.labler(label=label)
        # TODO: upload the count to the server

    images.stop()


videoprocessor()
