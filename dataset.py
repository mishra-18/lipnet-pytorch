import cv2
import torch
from torch.utils.data import Dataset
from utils import get_stoi
import numpy as np
import os

class CustomDataset(Dataset):
  def __init__(self , files):
    self.files = files

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    path = self.files[idx]
    vocab = get_stoi(path)
    mpgpath = os.getcwd() + "/data/s1/"
    mpgpath = mpgpath + path.split("/")[5].split(".")[0] + ".mpg"
    frames = []
    cap = cv2.VideoCapture(mpgpath)
    ret = True
    size = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while ret:
      ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
      if ret:
         img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
         img = np.reshape(img , ( img.shape[0] , img.shape[1] , 1 ))
         frames.append(img[190:236, 80:220, :])

    for _ in range(75-int(size)):
        frames.append(np.zeros((46, 140, 1)))
    mpg = np.stack(frames, axis=0)
    frames = torch.from_numpy(mpg)
    frames = torch.permute(frames , (3 , 0 , 1 , 2))
    return frames , vocab