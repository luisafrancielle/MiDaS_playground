#dependencias
import cv2
import torch
import matplotlib.pyplot as plt

#download the model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

#input transformational pipeline
transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transform.small_transform

#put in cv
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    #make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()
    
    plt.imshow(output)

    cv2.imshow('CV2frame',frame)
    plt.pause(0.000001)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

plt.show()