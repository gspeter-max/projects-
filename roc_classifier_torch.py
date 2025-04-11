import torch 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.preprocessing import StandardScaler 

x, y  = load_breast_cancer(return_X_y= True) 

sss = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, random_state= 42) 
train_idx , test_idx = next(sss.split(x,y)) 
x_train , x_test = x[train_idx],  x[test_idx] 
y_train , y_test = y[train_idx] , y[test_idx] 
print(x_train)

scaler = StandardScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = torch.tensor(x_train, dtype = torch.float32) 
x_test = torch.tensor(x_test, dtype = torch.float32) 
y_train = torch.tensor(y_train, dtype = torch.float32) 
y_test = torch.tensor(y_test, dtype = torch.float32) 


class auc_classification(torch.nn.Module): 
    def __init__(self,input_dim): 
        super().__init__() 
        self.layer1 = torch.nn.Linear(input_dim, 64) 
        self.layer2 = torch.nn.Linear(64, 32) 
        self.layer3 = torch.nn.Linear(32,1) 
    
    def forward(self,inputs):
        x  = torch.relu(self.layer1(inputs)) 
        x = torch.relu(self.layer2(x))
        return self.layer3(x) 

def custom_loss(y_true, y_pred): 
    '''
    torch.log(1 + torch.exp(-diff)) ===  torch.nn.functional.softplus(-dff) 
    
    '''
    pos_class = y_pred[y_true == 1] 
    neg_class = y_pred[y_true == 0]
    
    num_pos = pos_class.numel() 
    num_neg = neg_class.numel() 
    
    if num_pos == 0 or num_neg == 0: 
        return torch.tensor(0.0,requires_grad=True) 
    
    pos_class = pos_class.unsqueeze(1) 
    neg_class = neg_class.unsqueeze(0) 
    
    diff = pos_class - neg_class 
    loss = torch.nn.functional.softplus(-diff) 
    return torch.mean(loss) 


model = auc_classification(x_train.shape[1]) 
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001) 

for i in range(100): 
    optimizer.zero_grad() 
    output = model(x_train).squeeze() 
    loss = custom_loss(y_train,output)
    loss.backward() 
    optimizer.step() 
    
    if i % 10 == 0: 
        print(f'epoch - {i}  --  loss  - {loss}')

y_pred = model(x_test)

from sklearn.metrics import roc_auc_score
# from metrics.metrics import my_roc_auc_score
# my_roc = my_roc_auc_score(y_test.detach().numpy(), y_pred.detach().numpy())
roc = roc_auc_score(y_test.detach().numpy(), y_pred.detach().numpy())
# print(f' my roc function is : {my_roc}')
print(f' sklearn  roc function is : {roc}')
