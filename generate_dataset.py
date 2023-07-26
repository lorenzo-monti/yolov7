import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import json 
import time

from pose import Pose
import argparse
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
      image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
      output, _ = model(image)
    return output, image




def draw_keypoints(output, image, t, resolution_vid, list_keypoints):
  
  output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
  with torch.no_grad():
        output = output_to_keypoint(output)
  nimg = image[0].permute(1, 2, 0) * 255
  nimg = nimg.cpu().numpy().astype(np.uint8)
  nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

  for idx in range(output.shape[0]):

      if(len(output) == 1) :

        output = output[idx, 7:].T
        plot_skeleton_kpts(nimg, output, 3)
    
        cpt_id = 0  
        for i in range (len(output)) :
            if i%3 == 0 :

                x = (output[i]*resolution_vid[0])/(nimg.shape[1])
                y = (output[i+1]*resolution_vid[1])/(nimg.shape[0])
            
                list_keypoints.extend([[cpt_id, t+1, x, y]])
                cpt_id += 1

  return nimg, list_keypoints
def pose_estimation_video(filename):
    list_keypoints = []

    cap = cv2.VideoCapture(filename)

    resolution_vid = [cap.get(3), cap.get(4)]

    num_of_timestep = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow('Pose estimation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose estimation', 1440, 810)

    for t in range(num_of_timestep) :
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, frame = run_inference(frame)
            frame, list_keypoints = draw_keypoints(output, frame, t, resolution_vid, list_keypoints)
            
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
        
            cv2.imshow('Pose estimation', frame)
        else:
            break

        k = cv2.waitKey(30)
        if k==27:    # Esc key to stop
            cap.release()
            cv2.destroyAllWindows()
            break 
        if k==32: #espace key to pause
            cv2.waitKey(0)

    cap.release()
    
    cv2.destroyAllWindows()

    return list_keypoints

def export_data_frame(data,name) : 
    '''Export the dataframe in a json file'''
    list_pose = []

    TimeStamp = data["TimeStamp"].astype(int)
    data['coordx'] = data['coordx'].astype(float)
    data['coordy'] = data['coordy'].astype(float)

    max_timestamp = max(data["TimeStamp"])

    for i in range (1,max_timestamp+1) :

        
        df_pose = data.loc[TimeStamp == i]
        human = []
        for index, row  in df_pose.iterrows():
            PartId = int(row['PartId'])
            coordx = row['coordx']
            coordy = row['coordy']
  
            bodypart = Pose.BodyPart(PartId,coordx,coordy)
            human.append(bodypart)

        pose_t = Pose.Pose()
        pose_t.set_time_stamp(i)
        pose_t.set_body_parts(human)
        list_pose.append(pose_t)
        

    dictOfPoses = [list_pose[i].as_json() for i in range(0, len(list_pose) ) ]
        
    with open(name+'.json', 'w', encoding='utf-8') as f:
        json.dump(dictOfPoses, f, ensure_ascii=False, indent=4)


def main():
    os.chdir(r"/media/lorenzo/easystore/20230306_Cephismer")

    list_dir = os.listdir("./dataset")
    list_dir = sorted(list_dir)


    for i in range (8, len(list_dir)) :


        dir = list_dir[i]
        list_vid = os.listdir("./dataset/"+dir+"/vid")
        list_vid = sorted(list_vid)

        for j in range (len(list_vid)) :
            vid = list_vid[j]
            
            # if(vid.split('_')[2]=='wat') :
            name_vid = 'xy_rawy7'+ vid[9:-4]

            list_keypoints = pose_estimation_video("./dataset/"+dir+"/vid/"+vid)

            df_yolov7 = pd.DataFrame(list_keypoints, columns=['PartId','TimeStamp','coordx','coordy'])
            df_yolov7 = df_yolov7.sort_values(by = ['TimeStamp','PartId']).reset_index(drop=True)

            print(vid)

            export_data_frame(df_yolov7, "./dataset/"+dir+"/y7/"+name_vid)

            
                
            # else :
            #     name_vid = 'xy_rawy7'+ vid[8:-4]
            

if __name__ == "__main__":
    main()

