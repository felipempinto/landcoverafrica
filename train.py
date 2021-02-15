import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from models.unet import unet
from models.unet2 import UNet
from utils.dataloader import *
import matplotlib.pyplot as plt
import cv2

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

def pixel_acc(pred, label,img):
    _, preds = torch.max(pred, dim=1)
    # for i,j,im in zip(preds,label,img):
    #     f, axarr = plt.subplots(1,3)
    #     axarr[0].set_title("Original")
    #     axarr[1].set_title("Predicted")
    #     axarr[2].set_title("Label")
    #     axarr[0].imshow(cv2.merge(im.cpu().detach().numpy()))
    #     axarr[1].imshow(i.cpu().detach().numpy())
    #     axarr[2].imshow(j.cpu().detach().numpy())
    #     plt.show()

    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc.cpu().detach().numpy()


class Trainer:
    global dev
    device = dev

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 training_Dataloader,
                 validation_Dataloader = None,
                 lr_scheduler = None,
                 epochs=100):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_Dataloader = training_Dataloader
        self.validation_Dataloader = validation_Dataloader 
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs

        self.training_loss = []
        self.training_acc = []
        self.validation_loss = []
        self.validation_acc = []
        self.learning_rate = []

    def run_trainer(self):
        progressbar = trange(self.epochs, desc='Progress')

        epoch = 0
        for i in progressbar:
            epoch+=1
            self._train()

            if self.validation_Dataloader is not None:
                self._validate()

            if self.lr_scheduler is not None:
                if self.validation_Dataloader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i]) 
                else:
                    self.lr_scheduler.batch()
        
        return self.training_loss, self.validation_loss, self.learning_rate,self.training_acc,self.validation_acc
    
    def _train(self):

        self.model.train()
        train_losses = []
        train_acc = []
        batch_iter = tqdm(enumerate(self.training_Dataloader),"Training",total = len(self.training_Dataloader),leave = False)
        
        for i,(x,y) in batch_iter:

            train_X,train_y = x.to(self.device),y.to(self.device)
            out = self.model(train_X)
            loss = self.criterion(out,train_y)
            acc = pixel_acc(out,train_y,train_X)#pred, label)
            loss_value = loss.item()
            train_losses.append(loss_value)
            train_acc.append(acc)
            loss.backward()
            self.optimizer.step()

            batch_iter.set_description(f"Training: (loss {loss_value:.4f} acc {acc:.4f})")
        self.training_loss.append(np.mean(train_losses))
        self.training_acc.append(np.mean(train_acc))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()
    
    def _validate(self):

        self.model.eval()
        valid_losses = []
        valid_acc = []
        batch_iter = tqdm(enumerate(self.validation_Dataloader),"Validation",total = len(self.validation_Dataloader),leave=False)

        for i, (x,y) in batch_iter:
            test_X,test_y = x.to(self.device),y.to(self.device)

            with torch.no_grad():
                out = self.model(test_X)
                loss = self.criterion(out,test_y)
                acc = pixel_acc(out,test_y)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                valid_acc.append(acc)
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f} / acc {acc:.4f})')
        
        self.validation_loss.append(np.mean(valid_losses))
        self.validation_acc.append(np.mean(valid_acc))
        batch_iter.close()



if __name__=="__main__":


    BATCH_SIZE = 10
    X = get_file_names('dataset/inputs')
    y = get_file_names('dataset/target')

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.25,shuffle=True)

    dataset_train = DataSet(inputs = X_train, targets = y_train)
    dataset_valid = DataSet(inputs = X_test, targets = y_test)

    dataloader_training = DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True)
    dataloader_validation = DataLoader(dataset=dataset_valid,batch_size=BATCH_SIZE,shuffle=True)

    n_of_classes = 8
    model = unet(dimensions=n_of_classes).to(dev)
    # model = UNet(in_channels = 3,
    #              out_channels = n_of_classes
    #              ).to(dev)
    
    criterion = torch.nn.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-1).cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.005)

    trainer = Trainer(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_Dataloader=dataloader_training,
                  validation_Dataloader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=10)

    training_losses, validation_losses, lr_rates,training_acc,validation_acc = trainer.run_trainer()

    times = [i+1 for i in range(len(training_losses))]
    fig = plt.figure()
    
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1),(1,0),sharex=ax1)

    ax1.plot(times,training_acc,label = 'acc')
    ax1.plot(times,validation_acc,label = 'val_acc')
    ax1.legend(loc=2)

    ax2.plot(times,training_losses,label = 'loss')
    ax2.plot(times,validation_losses,label = 'val_loss')
    ax2.legend(loc=2)

    plt.show()
