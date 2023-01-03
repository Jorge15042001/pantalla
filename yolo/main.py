import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


#  import matplotlib.pyplot as plt
import cv2
import numpy as np

from map_3d import drawSurface, drawPoints, norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def load_model():
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model


model = load_model()


def run_inference(image):
    # Resize and pad image
    image, ratio, padding = letterbox(
        image, 960, stride=64, auto=True)  # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image)  # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
        image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0)  # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    return output, image, ratio, padding


def draw_keypoints(output, image, ratio, padding):
    output = non_max_suppression_kpt(output,
                                     0.25,  # Confidence Threshold
                                     0.65,  # IoU Threshold
                                     nc=model.yaml['nc'],  # Number of Classes
                                     # Number of Keypoints
                                     nkpt=model.yaml['nkpt'],
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        k_pts = output[idx, 7:].T
        eye1x = int(k_pts[3*1])
        eye1y = int(k_pts[3*1+1])
        conf_eye1 = k_pts[3*1+2]
        eye2x = int(k_pts[3*2])
        eye2y = int(k_pts[3*2+1])
        conf_eye2 = k_pts[3*2+2]
        eye = np.array(((eye1x+eye2x)/2, (eye1y+eye2y)/2), np.float32)
        #  cv2.circle(nimg, eye.astype(int), 20, (255,255,255), -1)
        valid_eye = conf_eye1 > 0.5 and conf_eye2 > 0.5

        feet1x = int(k_pts[3*15])
        feet1y = int(k_pts[3*15+1])
        conf_feet1 = k_pts[3*15+2]
        feet2x = int(k_pts[3*16])
        feet2y = int(k_pts[3*16+1])
        conf_feet2 = k_pts[3*16+2]
        feet = np.array(((feet1x+feet2x)/2, (feet1y+feet2y)/2), np.float32)
        valid_feet = conf_feet1 > 0.5 and conf_feet2 > 0.5

        shoulder1x = int(k_pts[3*5])
        shoulder1y = int(k_pts[3*5+1])
        conf_shoulder1 = k_pts[3*5+2]
        shoulder2x = int(k_pts[3*6])
        shoulder2y = int(k_pts[3*6+1])
        conf_shoulder2 = k_pts[3*6+2]
        shoulder_dist = norm(shoulder1x,shoulder1y,shoulder2x,shoulder1y)
        valid_shoulder = conf_shoulder1 > 0.5 and conf_shoulder2 > 0.5



        if valid_eye and valid_feet:
            #  drawPoints(nimg, (eye, feet), ratio, padding)
            pass
        if valid_eye and valid_shoulder:
            head = shoulder_dist/2
            feet = eye+np.array((0,eye[1]+7.5*head),float)
            drawPoints(nimg,(eye,feet),ratio,padding)


    return nimg


def pose_estimation_webcam():
    cap = cv2.VideoCapture(0)
    # VideoWriter for saving the video
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, frame, ratio, padding = run_inference(frame)
            frame = draw_keypoints(output, frame, ratio, padding)
            drawSurface(frame, ratio, padding)
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow('Pose estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


pose_estimation_webcam()
