import torch
import cv2
from time import time
import numpy as np
from ultralytics import YOLO
import supervision as sv

class ObjectID:
    def __init__(self , capture_index):
        self.captureIndex = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: " + self.device)

        self.model = self.loadModel()

        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    
    def loadModel(self):
        model = YOLO('yolov8n.pt') # Cambiar el modelo ya entrenado
        model.fuse()
        
        return model
    
    def predict (self , frame):
        results = self.model(frame)
        return results
    
    def plotBox (self , results , frame):
        xyxys = []
        data = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy
            if boxes:
                class_id = boxes.cls[0]

                if class_id == 0.0:
                    for xyxy in xyxys:
                        cv2.rectangle(frame , (int(xyxy[0]) , int(xyxy[1])) , (int(xyxy[2]) , int(xyxy[3])) , (0 , 255 , 0) , 3)
                        result , res = self.data_find(frame , xyxy)
                        data.append(res)

            
            else: print("No se detectan objetos")
            
        return frame , data
    
    def data_find(self ,frame , xyxy):
        img = frame[int(xyxy[1]):int(xyxy[3]) , int(xyxy[0]):int(xyxy[2])]
        data = []
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        tet , th = cv2.threshold(gray , 200 , 255 , cv2.THRESH_BINARY)
        contornos , jerarquia = cv2.findContours(th , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

        """m = cv2.moments(contornos[0])
        ## Centroide ###
        if not (int(m["m00"]) & int(m["m00"] ) == 0):
            cx = int(m["m10"])/int(m["m00"])
            cy = int(m["m01"])/int(m["m00"])

        cv2.circle(th , (cx , cy) , 5 , (0 , 255 , 0) , -1)

        ## Orientacion ###
        mu20 = m["mu20"]
        mu11 = m["mu11"]
        mu02 = m["mu02"]

        theta = np.degrees(1/2 * np.arctan2(2 * mu11 , mu20 - mu02))

        data = [cx , cy , theta]"""

        return th , data

    
    def __call__ (self):
        cap = cv2.VideoCapture(self.captureIndex)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame , objects = self.plotBox(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectID(capture_index=1)
detector()