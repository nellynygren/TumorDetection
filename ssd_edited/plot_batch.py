from utils import *
from datasets import BrainDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision.transforms as T

from PIL import Image
from PIL import ImageOps


# Parameters
data_folder = '/home/stud/n/nelnyg22/TumorDetection/tumor_detect/input/brain-tumor-object-detection-datasets/axial_t1wce_2_class/'  # folder with data files
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 5
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
checkpoint = './checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint,map_location=torch.device('cpu'))
model = checkpoint['model']
model = model.to(device)

test_dataset = BrainDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult) 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=test_dataset.collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate function here



def plot_batch(test_loader,model):
    model.eval()
    images, boxes, labels, difficulties = next(iter(test_loader))
    #print(labels)

    images = images.to(device)  # (N, 3, 300, 300)
    #print(images.shape)

    # Forward prop.
    predicted_locs, predicted_scores = model(images)

    # Detect objects in SSD output
    det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                               min_score=0.8, max_overlap=0.99,
                                                                               top_k=1)
    # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

    # Store this batch's results for mAP calculation
    boxes = [b.to("cpu") for b in boxes]
    labels = [l.to("cpu") for l in labels]
    difficulties = [d.to("cpu") for d in difficulties]
    images = images.to("cpu")
    transf = T.ToPILImage()

    for i in range(0,batch_size):
        ax=plt.subplot(2,5,i+1)
        im = images[i]
        im = transf(im)
        bb = boxes[i]*im.size[-1]
        #print(im.size)
        bb = bb.detach().numpy()
        #print(bb)

        # Plot image and bounding box
        plt.axis('off')
        im = ImageOps.invert(ImageOps.grayscale(im))
        plt.imshow(im, cmap='gray')
        for b_i in range(0,bb.shape[0]):
            # Create a Rectangle patch
            bb_sz_x = bb[b_i,2] - bb[b_i,0]
            bb_sz_y = bb[b_i,3] - bb[b_i,1]
            bb_pos_x = bb[b_i,0] - bb_sz_x/2 # center of box
            bb_pos_y = bb[b_i,1] - bb_sz_y/2 # center of box
            bb_plot = patches.Rectangle((bb_pos_y, bb_pos_x), bb_sz_y, bb_sz_x, linewidth=1, edgecolor='g', facecolor='None')
            # Add the patch to the Axes
            #print(bb_plot)
            ax.add_patch(bb_plot)
            
    boxes = [b.to("cpu") for b in det_boxes_batch]
    labels = [l.to("cpu") for l in det_labels_batch]
    difficulties = [d.to("cpu") for d in difficulties]
    #images = images.to("cpu")
    #transf = T.ToPILImage()

    for i in range(0,batch_size):
        ax=plt.subplot(2,5,i+1)
        im = images[i]
        im = transf(im)
        bb = boxes[i]*im.size[-1]
        #print(im.size)
        bb = bb.detach().numpy()

        # Plot image and bounding box
        plt.axis('off')
       # plt.imshow(im, cmap='gray')
        for b_i in range(0,bb.shape[0]):
            # Create a Rectangle patch
            bb_sz_x = bb[b_i,2] - bb[b_i,0]
            bb_sz_y = bb[b_i,3] - bb[b_i,1]
            bb_pos_x = bb[b_i,0] - bb_sz_x/2 # center of box
            bb_pos_y = bb[b_i,1] - bb_sz_y/2 # center of box
            bb_plot = patches.Rectangle((bb_pos_y, bb_pos_x), bb_sz_y, bb_sz_x, linewidth=1, edgecolor='r', facecolor='None')
            # Add the patch to the Axes
            ax.add_patch(bb_plot)
        conf_score = det_scores_batch[0].item()
        class_pred = det_labels_batch[0].item()
        ax.set_title('Conf: {0:.1f}, class: {1:.0f}'.format(conf_score,class_pred),fontsize=7)

    plt.savefig('./figures/plot_test_batch')
    
if __name__ == '__main__':
    plot_batch(test_loader, model)