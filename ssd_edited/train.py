import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import BrainDataset, PascalVOCDataset
from utils import *
from torch.utils.tensorboard import SummaryWriter

thresh = 0.5
negpos = 3
alpha_loc = 10



writer = SummaryWriter('runs/lr1e4_bs20_thr05_negpos3_alpha10_newacc')


# Data parameters
data_folder = '/home/stud/n/nelnyg22/TumorDetection/tumor_detect/input/brain-tumor-object-detection-datasets/axial_t1wce_2_class/'  # folder with data files
keep_difficult = False  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 2 #len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Learning parameters
checkpoint = None #'./checkpoint_ssd300.pth.tar'  # path to model checkpoint, None if none
batch_size = 20  # batch size
val_batch_size = 10

iterations = 100000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 100  # print training status every __ batches
lr = 1e-4  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy,threshold = thresh,neg_pos_ratio = negpos,alpha=alpha_loc).to(device)

    # Custom dataloaders
    # train_dataset = PascalVOCDataset(data_folder,
    #                                  split='train',
    #                                  keep_difficult=keep_difficult)
    train_dataset = BrainDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)        
    validation_dataset = BrainDataset(data_folder,
                                     split='val',
                                     keep_difficult=keep_difficult) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=14, shuffle=True,
                                               collate_fn=validation_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    len_validation_dataset = len(validation_dataset)

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              validation_loader=validation_loader,
              len_validation_dataset = len_validation_dataset)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)
        print("Saved checkpoint")


def train(train_loader, model, criterion, optimizer, epoch, validation_loader, len_validation_dataset):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    
    start = time.time()
    
    # Batches
    for i, (images, boxes, labels, difficulties) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)
                
        start = time.time()
        
    # AFTER LAST BATCH OF EACH EPOCH:
    # Evaluate
    model.eval()
    # Calculate and best box accuracy for this training batch
    det_boxes_best, det_labels_best, det_scores_best = model.detect_objects(predicted_locs, predicted_scores,
                                                                           min_score=0.01, max_overlap=0.45,
                                                                           top_k=1)

    train_calculate_best_box_accuracy = calculate_best_box_accuracy(det_boxes_best, det_labels_best, det_scores_best, boxes, labels)
    train_bb_acc = torch.mean(train_calculate_best_box_accuracy)
            
    # Write training loss and accuracy
    writer.add_scalar('Loss/train',losses.avg,epoch)
    writer.add_scalar('Accuracy/best_box_train',train_bb_acc,epoch)
    
    del predicted_locs, predicted_scores, det_boxes_best, det_labels_best, det_scores_best

            
    # Evaluate for entire validation dataset
    val_bb_acc = AverageMeter()
    val_APs = AverageMeter()
    val_mAP = AverageMeter()
    val_loss = AverageMeter()
    for v in [1]: #, (images_val, boxes_val, labels_val, difficulties_val) in enumerate(validation_loader):
        images_val, boxes_val, labels_val, difficulties_val = next(iter(validation_loader))
        # Move to default device
        images_val = images_val.to(device)  # (batch_size (N), 3, 300, 300)
        boxes_val = [b.to(device) for b in boxes_val]
        labels_val = [l.to(device) for l in labels_val]
        difficulties_val = [d.to(device) for d in difficulties_val]

        model.eval()
        predicted_locs_val, predicted_scores_val = model(images_val)

        # Calculate loss for this batch
        val_loss.update(criterion(predicted_locs_val, predicted_scores_val, boxes_val, labels_val))  # scalar

        # Calculate accuracy metric for this batch
        model.eval()
        val_det_boxes_batch, val_det_labels_batch, val_det_scores_batch = model.detect_objects(predicted_locs_val, predicted_scores_val,
                                                                               min_score=0.01, max_overlap=0.45,
                                                                               top_k=200)
        val_APs_batch, val_mAP_batch = calculate_mAP(val_det_boxes_batch, val_det_labels_batch, val_det_scores_batch, boxes_val, labels_val, difficulties_val)
        
        val_mAP.update(val_mAP_batch)

        # Calculate best box accuracy
        model.eval()
        val_det_boxes_best, val_det_labels_best, val_det_scores_best = model.detect_objects(predicted_locs_val, predicted_scores_val,
                                                                               min_score=0.01, max_overlap=0.45,
                                                                               top_k=1)

        val_bb_acc.update(torch.mean(calculate_best_box_accuracy(val_det_boxes_best, val_det_labels_best, val_det_scores_best, boxes_val, labels_val)))

        del predicted_locs_val, predicted_scores_val,images_val, boxes_val, labels_val, val_det_boxes_best, val_det_labels_best, val_det_scores_best
        
    # Write validation loss and validation accuracy
    writer.add_scalar('Loss/validation',val_loss.avg,epoch)
    writer.add_scalar('Accuracy/mAP_validation',val_mAP.avg,epoch)
    writer.add_scalar('Accuracy/best_box_validation',val_bb_acc.avg,epoch)

    # Print status
    
    print('Epoch: [{0}][{1}/{2}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Val loss {val_loss:.4f}\t'
          'MAP: {maps:.2f}'.format(epoch, i, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time, loss=losses,val_loss=val_loss.avg,maps=val_mAP.avg))
    
    model.train()
    
    # free some memory since their histories may be stored
    
    del images, boxes, labels
    


if __name__ == '__main__':
    main()
