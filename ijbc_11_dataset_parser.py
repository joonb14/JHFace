import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm
from pathlib import Path

enroll_df = pd.read_csv("/raid/workspace/jihyun/IJBC/IJB/IJB-C/protocols/test1/enroll_templates.csv")
verif_df = pd.read_csv("/raid/workspace/jihyun/IJBC/IJB/IJB-C/protocols/test1/verif_templates.csv")
image_df = pd.concat([enroll_df, verif_df])

face_x_list = image_df["FACE_X"].tolist()
face_y_list = image_df["FACE_Y"].tolist()
face_width_list = image_df["FACE_WIDTH"].tolist()
face_height_list = image_df["FACE_HEIGHT"].tolist()
files = image_df["FILENAME"].tolist()

for i, file in enumerate(tqdm(files)):
    Path("/raid/workspace/jbpark/IJB-C/veri_crops/").mkdir(parents=True, exist_ok=True)
    img = cv2.imread("/raid/workspace/jihyun/IJBC/IJB/IJB-C/images/"+files[i])
    crop_img = img[face_y_list[i]:face_y_list[i]+face_height_list[i], face_x_list[i]:face_x_list[i]+face_width_list[i]]
    cv2.imwrite("/raid/workspace/jbpark/IJB-C/veri_crops/"+str(i+1)+'.jpg',crop_img)