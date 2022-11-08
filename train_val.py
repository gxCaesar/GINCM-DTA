from utils import *
from model import *
from dataset import *
from torch_geometric.loader import DataLoader
import time
import datetime
from tensorboardX import SummaryWriter

def train_eval(model, optimizer,scheduler, train_loader, valid_loader,test_loader, epochs=2 , log_path = 'default'):
    path = './model/' + log_path + '.pt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('-----Training-----')
    starttime = datetime.datetime.now()
    last_epoch_time = starttime

    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_test_epoch = -1
    for epoch in range(epochs):
        endtime = datetime.datetime.now()
        print('total run time: ', endtime - starttime)
        print('last epoch run time: ', endtime - last_epoch_time)
        last_epoch_time = endtime
        print('Epoch', epoch)
        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)
        val1 = get_mse(G, P)
        if test_loader is not None:
            print('predicting for test data')
            G, P = predicting(model, device, test_loader)
            val2 = get_mse(G, P)
            if val2 < best_test_mse:
                best_test_mse = val2
                best_test_epoch = epoch + 1
                print('test mse has improved at epoch ', best_test_epoch, "test mse:", best_test_mse)
        if val1 < best_mse:
            best_mse = val1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), path)
            if test_loader is not None:
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse, "test mse:", val2)
            else:
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
        else:
            if test_loader is not None:
                print('current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse,
                      "Best test at:", best_test_epoch, '; best_test_mse', best_test_mse)
            else:
                print('current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse)
        scheduler.step()
        #print(optimizer.state_dict()['param_groups'][0]['lr'])

