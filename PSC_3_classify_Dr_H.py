import sys
sys.path.append(r'I:\hjj\OneDrive\saratr\Pyg code\2d__PSC_SARATR_soc_3_H_brdm_s1_zsu_17&15\libs')
import shutil
import torch
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, FiLMConv
from torch_geometric.nn import TopKPooling, ASAPooling
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from dataset_build_dr_H import TrainDataset, TestDataset
# from torch.utils.data import ConcatDataset
import numpy as np
import os
import win32file

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

def deletedir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

deletedir(r'I:\hjj\OneDrive\saratr\Pyg code\2d__PSC_SARATR_soc_3_H_brdm_s1_zsu_17&15\mydata\TestDataset')
deletedir(r'I:\hjj\OneDrive\saratr\Pyg code\2d__PSC_SARATR_soc_3_H_brdm_s1_zsu_17&15\mydata\TrainDataset')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
hidden_layer = 512

# model path
model_path = r'./model/pyg.pth'

# 加载数据、拆分数据集
traindata = TrainDataset("./mydata/TrainDataset") # 创建训练数据集对象
testdata = TestDataset("./mydata/TestDataset") # 创建训练数据集对象
test_loader = DataLoader(testdata, batch_size = 300, shuffle = True, drop_last = False)
train_loader = DataLoader(traindata, batch_size = 20, shuffle = True, drop_last = False)

def is_used(file_name):
    try:
        vHandle = win32file.CreateFile(file_name, win32file.GENERIC_READ, 0, None, win32file.OPEN_EXISTING, win32file.FILE_ATTRIBUTE_NORMAL, None)
        return int(vHandle) == win32file.INVALID_HANDLE_VALUE
    except:
        return True
    finally:
        try:
            win32file.CloseHandle(vHandle)
        except:
            pass

# 构建模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()  
        self.conv1 = FiLMConv(traindata.num_features, hidden_layer)
        self.conv2 = FiLMConv(hidden_layer, hidden_layer)
        self.conv3 = FiLMConv(hidden_layer, hidden_layer)
        # self.conv4 = FiLMConv(hidden_layer, hidden_layer, act=None)
        # self.conv5 = FiLMConv(hidden_layer, hidden_layer, act=None)

        self.GCN_BN = BatchNorm(hidden_layer)
        self.bn = torch.nn.BatchNorm1d(hidden_layer)
        self.pool = ASAPooling(hidden_layer)

        self.lin1 = torch.nn.Linear(hidden_layer * 2, hidden_layer)
        # self.lin2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # self.lin3 = torch.nn.Linear(hidden_layer, hidden_layer)
        # self.lin2 = torch.nn.Linear(hidden_layer, hidden_layer)
        self.classify = torch.nn.Linear(hidden_layer, traindata.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x1 = F.elu(self.GCN_BN(self.conv1(x, edge_index, edge_attr)))
        x2 = F.elu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.elu(self.conv3(x2, edge_index, edge_attr))
        # x4 = F.elu(self.conv4(x3, edge_index, edge_attr))
        # x5 = F.elu(self.conv5(x4, edge_index, edge_attr))
        # x6 = F.elu(self.conv6(x5, edge_index, edge_attr))
        # x = torch.cat((x1, x2, x3), dim=1)
        # x_pool, edge_index, edge_attr, batch, _ = self.pool(x3, edge_index, edge_attr, batch)
        
        # readout1 = global_add_pool(x3, batch)
        readout1 = torch.cat([global_mean_pool(x3, batch), global_max_pool(x3, batch)], dim=1)
        # readout2 = torch.cat([global_mean_pool(x2, batch), global_max_pool(x2, batch)], dim=1)
        # readout3 = torch.cat([self.GCN_BN(global_mean_pool(x3, batch)), global_max_pool(x3, batch)], dim=1)
        readout = readout1
        # x = F.elu(self.bn(F.dropout(self.lin1(readout),0.7)))
        x = F.elu(self.bn(self.lin1(readout)))
        # x = F.elu(self.bn(self.lin2(x)))
        # x = F.elu(self.bn(self.lin3(x)))
        x = self.classify(x)
        return x
    
def init_weights(self):
    for layer in self.modules():
        if isinstance(layer, torch.nn.Linear):
            init.zeros_(layer.weight.data)
            if layer.bias is not None:
                init.zeros_(layer.bias.data)
			    
# 训练与评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# model.apply(init_weights)
# model.load_state_dict(torch.load(model_path))
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=0.1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.25)

def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        label = []

        for i in range(len(data.y)):
            if data.y[i] == 0:
                temp = [1.0, 0.0, 0.0]
            elif data.y[i] == 1:
                temp = [0.0, 1.0, 0.0]
            elif data.y[i] == 2:
                temp = [0.0, 0.0, 1.0]
            # elif data.y[i] == 3:
            #     temp = [0.0, 0.0, 0.0, 1.0]
            # elif data.y[i] == 4:
            #     temp = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # elif data.y[i] == 5:
            #     temp = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            # elif data.y[i] == 6:
            #     temp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            # elif data.y[i] == 7:
            #     temp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            # elif data.y[i] == 8:
            #     temp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            # elif data.y[i] == 9:
            #     temp = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                
            label.append(temp)
        label = torch.from_numpy(np.array(label))
        label = label.float().to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label, label_smoothing = 0.05)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    # scheduler.step()
    # print(optimizer.param_groups[0]["lr"])
    
    return loss_all / len(traindata)

def test1(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        idx = pred.eq(data.y)
        correct += idx.sum().item()
        
    return correct/len(loader.dataset)

def test2(loader):
    model.eval()
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0 
    # correct4 = 0
    # correct5 = 0
    # correct6 = 0 
    # correct7 = 0
    # correct8 = 0
    # correct9 = 0 
    # correct10 = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        idx = pred.eq(data.y)
        correct += idx.sum().item()
        temp = pred[idx].cpu().numpy()
        correct1 += len(np.argwhere(temp == 0))
        correct2 += len(np.argwhere(temp == 1))
        correct3 += len(np.argwhere(temp == 2))
        # correct4 += len(np.argwhere(temp == 3))
        # correct5 += len(np.argwhere(temp == 4))
        # correct6 += len(np.argwhere(temp == 5))
        # correct7 += len(np.argwhere(temp == 6))
        # correct8 += len(np.argwhere(temp == 7))
        # correct9 += len(np.argwhere(temp == 8))
        # correct10 += len(np.argwhere(temp == 9))
        
    return correct/len(loader.dataset), correct1, correct2, correct3

# acclist = []
maxacc = 0
maxc1 = 0
maxc2 = 0
maxc3 = 0
for epoch in range(1, 5000):
    loss = train(epoch)
    train_acc = test1(train_loader)
    test_acc, c1, c2, c3 = test2(test_loader)
    if test_acc > maxacc:
        maxacc = test_acc
        maxc1 = c1
        maxc2 = c2
        maxc3 = c3
        # if not is_used(model_path):
        #     try:
        #         print('sava model')
        #         torch.save(model.state_dict(), model_path)
                
        #     except PermissionError:
        #         print('do except: PermissionError')
        #         pass
    # acclist.append(test_acc)

    # plt.cla()
    # plt.plot(np.linspace(1, len(acclist), len(acclist)), acclist)
    # plt.yscale('log')
    # plt.savefig('./test_acc.png')

    print('Epoch:{:03d}, Loss:{:.4f}, Train:{:.4f}, Test:{:.4f}, Max Acc:{:.4f}, brdm2:{:.0f}/274, 2s1:{:.0f}/274, zsu234:{:.0f}/274'.
          format(epoch, loss, train_acc, test_acc, maxacc, maxc1, maxc2, maxc3))