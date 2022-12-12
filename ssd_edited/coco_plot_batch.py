from utils import *
from datasets import COCODataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision
import torchvision.transforms as T

plt.rcParams['figure.figsize'] = [16, 8]

from PIL import Image
from PIL import ImageOps

from model import MultiBoxLoss

pred = True


# Parameters
data_folder = './coco_part2' #'/home/stud/n/nelnyg22/TumorDetection/tumor_detect/input/brain-tumor-object-detection-datasets/axial_t1wce_2_class/'  # folder with data files
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 10
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
checkpoint = './short_checkpoint_ssd300.pth.tar' #'./long_night_checkpoint_ssd300.pth.tar'

coco_dataset = COCODataset(data_folder,split='train',
                                         keep_difficult=keep_difficult) 
coco_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=coco_dataset.collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate function here
model = None

if pred == True:
    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint,map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.eval()
    model = model.to(device)

    thresh = 0.5
    negpos = 1
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy,threshold = thresh,neg_pos_ratio = negpos).to(device)


def plot_batch(test_loader,model):
    images, boxes, labels, difficulties = next(iter(test_loader))

    images = images.to(device)  # (N, 3, 300, 300)

    if pred == True:
        model.eval()
        # Forward prop.
        predicted_locs, predicted_scores = model(images)

        # Detect objects in SSD output
        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                   min_score=0.3, max_overlap=1,
                                                                                   top_k=10)
        # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

        #loss_batch = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

    # Store this batch's results for mAP calculation
    boxes = [b.to("cpu") for b in boxes]
    labels = [l.to("cpu") for l in labels]
    difficulties = [d.to("cpu") for d in difficulties]
    images = images.to("cpu")
    transf = T.ToPILImage()

    fillbox = dict(boxstyle='square', facecolor='g', alpha=0.5)

    for i in range(0,batch_size):
        ax=plt.subplot(2,5,i+1)
        im = images[i]
        im = transf(im)
        bb = boxes[i]#*im.size[-1]
        #print(im.size)
        bb = bb.detach().numpy()
        #print(bb)

        # Plot image and bounding box
        plt.axis('off')
        #im = ImageOps.invert(ImageOps.grayscale(im))
        plt.imshow(im)
        for b_i in range(0,bb.shape[0]):
            # Create a Rectangle patch
            bb_sz_x = bb[b_i,2] - bb[b_i,0]
            bb_sz_y = bb[b_i,3] - bb[b_i,1]
            bb_pos_x = bb[b_i,0]
            bb_pos_y = bb[b_i,1]
            bb_plot = patches.Rectangle((bb_pos_x*300, bb_pos_y*300), bb_sz_x*300, bb_sz_y*300, linewidth=1, edgecolor='g', facecolor='None')
            ax.text(bb_pos_x*300,bb_pos_y*300,'Plane, GT',fontsize=6,color='w',bbox=fillbox)
            
            #torchvision.utils.draw_bounding_box(im)
            #torchvision.utils.draw_bounding_boxes(
            #        (image*255).type(torch.uint8),
            #        annot["boxes"], 
            #        labels=list(map(str,labels)), 
            #        colors=list(Detection.__labels2rgb(labels, palette)),            # Add the patch to the Axes
            #print(bb_plot)
            ax.add_patch(bb_plot)
            
    if pred == True:
        ax=plt.subplot(2,5,i+1)
        det_boxes = [b.to("cpu") for b in det_boxes_batch]
        det_labels = [l.to("cpu") for l in det_labels_batch]
        det_difficulties = [d.to("cpu") for d in difficulties]
        #images = images.to("cpu")
        #transf = T.ToPILImage()
        #print(det_boxes)

        for i in range(0,batch_size):
            ax=plt.subplot(2,5,i+1)
            im = images[i]
            im = transf(im)
            bb = det_boxes[i]#*im.size[-1]
            #print(im.size)
            bb = bb.detach().numpy()

            # Plot image and bounding box
            plt.axis('off')
           # plt.imshow(im, cmap='gray')
        
            conf_scores = det_scores_batch[i].detach()
            for b_i in range(0,bb.shape[0]):
                # Create a Rectangle patch
                bb_sz_x = bb[b_i,2] - bb[b_i,0]
                bb_sz_y = bb[b_i,3] - bb[b_i,1]
                bb_pos_x = bb[b_i,0]
                bb_pos_y = bb[b_i,1]
                bb_plot = patches.Rectangle((bb_pos_x*300, bb_pos_y*300), bb_sz_x*300, bb_sz_y*300, linewidth=1, edgecolor='r', facecolor='None')
                # Add the patch to the Axes
                ax.add_patch(bb_plot)
                conf_score = conf_scores[b_i].item()
                ax.text(bb_pos_x*300,bb_pos_y*300,'Plane, {c:.2f}'.format(c=conf_score),fontsize=6,color='w',bbox=fillbox)
                
            #class_pred = det_labels_batch[i].item()
            #ax.set_title('Conf: {0:.1f}, class: {1:.1f}'.format(conf_score,class_pred),fontsize=7)
            #ax.set_title('Loss: {loss:.1f}'.format(loss=loss_batch.item()),fontsize=7)

            #det_boxes_best, det_labels_best, det_scores_best = model.detect_objects(predicted_locs, predicted_scores,
            #                                                                           min_score=0.01, max_overlap=0.45,
            #                                                                           top_k=1)


            #acc = calculate_best_box_accuracy(det_boxes_best, det_labels_best, det_scores_best, boxes, labels)
            #ax.set_title('Acc: {0:.2f}'.format(acc[0].item()),fontsize=7)

    plt.savefig('./figures/plot_test_batch')
    
if __name__ == '__main__':
    plot_batch(coco_loader, model)