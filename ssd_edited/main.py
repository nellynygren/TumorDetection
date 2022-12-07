import detect
from model import model
im = "/Users/nellynygren/Documents/DL/TumorDetection/tumor_detect/input/brain-tumor-object-detection-datasets/axial_t1wce_2_class/images/train"

#model.eval()
model(im,0,0.5,100)



