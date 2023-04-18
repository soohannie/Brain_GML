import os
import argparse
import time 
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter

from imports.ABIDEDataset import ABIDEDataset
from imports.utils import train_val_test_split

from models.gcn import GCN
from models.gat import GAT
from models.gatv2 import GATV2
from models.graphsage import GraphSAGE
from models.gin import GIN

def main(args):
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(os.path.join('./log', args.exp_name))

    # Dataset and Dataloader
    dataset = ABIDEDataset(root=args.data_path, name='ABIDE Dataset', transform=None, pre_transform=None)
    dataset.data.y = dataset.data.y.squeeze() 
    dataset.data.x[dataset.data.x == float('inf')] = 0
    train_idx, val_idx, test_idx = train_val_test_split(kfold=5, fold=0)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    train_loader = DataLoader(train_dataset,batch_size=args.bs, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    
    # Models
    # model_list = ['GCN', 'GAT', 'GRAPHSAGE','GIN']
    if args.model=='GCN':
        model = GCN(args.inp_dim, args.nclass, hidden_channels=32, dropout=0.5)
    elif args.model=='GAT':
        model = GAT(args.inp_dim, args.nclass, hidden_channels=32, n_heads=10, dropout=0.5)
    elif args.model=='GATV2':
        model = GATV2(args.inp_dim, args.nclass, hidden_channels=32, n_heads=2, dropout=0.6)
    elif args.model=='GRAPHSAGE':
        model = GraphSAGE(args.inp_dim, args.nclass, hidden_channels=32, dropout=0.5)
    elif args.model=='GIN':
        model = GIN(args.inp_dim, args.nclass, hidden_channels=32, dropout=0.5)

    model = model.to(device)
    
    #optimizer & Loss fn
    optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.weightdecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    
    if args.criterion=='CE':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion=='NLL':
        criterion = nn.NLLLoss()

    # Train and validate model
    start_time = time.time()
    print(f'Training Model {args.model}.')
    best_model = train(train_loader, val_loader, test_loader, optimizer, scheduler, model, criterion, device, writer, args)
    print(f'Finished Training. Time taken: {time.time()-start_time}')

    model.load_state_dict(best_model)
    model.eval()
    test_accuracy = test(test_loader, model, device)
    print("Test Acc: {:.7f} ".format(test_accuracy))

def train(train_loader, val_loader, test_loader, optimizer, scheduler, model, criterion, device, writer, args):
    best = 999999999
    for epoch in range(args.n_epochs):
        # Train ---------------------------------
        scheduler.step()
        step = 0
        training_loss = 0
        model.train()
        end = time.time()
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
            loss = criterion(out, data.y)
            writer.add_scalar('train_loss', loss, epoch*len(train_loader)+step)
            step = step + 1
            loss.backward()
            training_loss += loss.item() * data.num_graphs
            optimizer.step()

        # Validate -------------------------------
        model.eval()
        correct = 0
        val_loss = 0
        for data in val_loader:
            data = data.to(device)
            out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
            loss = criterion(out, data.y)
            writer.add_scalar('val_loss', loss, epoch*len(train_loader)+step)
            val_loss += loss.item() * data.num_graphs
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        val_accuracy = correct / len(val_loader.dataset)
        writer.add_scalar('val_acc', val_accuracy, epoch)

        # TEST ACCURACY IS CALCULATED only to observe the difference between val and test accuracies, and not actively used for picking the best performing model.
        # Only validation accuracy is used to pick the best model and save the model.
        correct = 0
        for data in test_loader:
            data = data.to(device)
            out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        test_accuracy = correct / len(test_loader.dataset)
        writer.add_scalar('test_acc', test_accuracy, epoch)

        if val_loss < best:
            best = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if args.save_model:
                torch.save(best_model_wts, os.path.join(args.save_path,args.exp_name+'.pth'))
        end = time.time() - end

        print('Epoch: [{epoch}]\t'
                'Time: {batch_time:.3f}\t'
                'Train Loss: {training_loss:.4f}\t'
                'Val Loss: {val_loss:.4f}\t'
                'Val Accuracy: {val_accuracy:.3f}\t'
                'Test Accuracy: {test_accuracy:.3f}\t'.format(
                epoch=epoch, batch_time=end, training_loss=training_loss, val_loss=val_loss, val_accuracy=val_accuracy, test_accuracy=test_accuracy))
        
        writer.add_scalars('Acc',{'val_acc':val_accuracy},  epoch)
        writer.add_scalars('Loss', {'train_loss': training_loss, 'val_loss': val_loss},  epoch)
    
    return best_model_wts
        
    
def test(test_loader, model, device):
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='fMRI GML Project')
    parser.add_argument('--data_path', type=str, default='/home/soohan/projects/Brain_GML/data/ABIDE_pcp/cpac/filt_noglobal')
    parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs of training')
    parser.add_argument('--bs', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
    parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
    parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
    parser.add_argument('--inp_dim', type=int, default=200, help='feature dim') # 200 ROIs -> 200 Dim
    parser.add_argument('--nroi', type=int, default=200, help='num of ROIs') # 200 Atlas ROIs
    parser.add_argument('--nclass', type=int, default=2, help='num of classes') # Binary Autism Disease Classification
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='path to save model')
    parser.add_argument('--model', type=str, default='GATV2', choices=['GCN', 'GAT', 'GATV2', 'GRAPHSAGE','GIN'], help='Choice of Model.')
    parser.add_argument('--criterion', type=str, default='NLL', choices=['NLL','CE'])
    parser.add_argument('--exp_name', type=str, default='GATV2')
    args = parser.parse_args()
    main(args)
