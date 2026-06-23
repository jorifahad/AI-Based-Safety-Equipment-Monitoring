from ultralytics import YOLO

model = YOLO('../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt')
results = model('../Model-Training/Dataset/test/images/ppe_1073_jpg.rf.72ea8a293a4f3e1135219e33701b1099.jpg') 
results[0].show()