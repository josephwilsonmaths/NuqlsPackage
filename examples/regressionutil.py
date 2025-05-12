import torch
import numpy as np

torch.set_default_dtype(torch.float64)

def toy_dataset(n_train,n_test,std):
    a1,a2,b1,b2 = -4,-2,2,4; c1,c2 = -6,6
    eps = torch.randn((n_train),1)*std
    train_x = torch.cat((torch.rand((int(n_train/2),1)) * (a2-a1) + a1, torch.rand((int(n_train/2),1)) * (b2-b1) + b1),dim=0)
    train_y = train_x**3 + eps
    test_x = torch.linspace(c1,c2,n_test).reshape(-1,1)
    test_y = torch.pow(test_x,3).reshape(-1,1)
    return train_x,train_y,test_x,test_y

class mlp(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output_size = 1
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 1)
        )

    def forward(self, x):
        output = self.net(x)
        return output
    
def train(X, y, model, loss_fn, optimizer, scheduler):
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return loss.item()

## Plot
def plot_torch(x):
    return x.detach().cpu().numpy().reshape(-1)

def plot_bayes_method(ax,mean,var,train_x,train_y,test_x,test_y,title,fs,ms,lw, legend_true=False):
    ax.plot(plot_torch(train_x),plot_torch(train_y),'ro',markersize=ms, label='Training data')
    ax.plot(plot_torch(test_x),plot_torch(test_y),'k',linewidth=lw, label='Target curve')
    ax.plot(plot_torch(test_x),plot_torch(mean),'b',linewidth=lw, label='Mean prediction')

    ci = torch.sqrt(var)*3
    y1, y2 = mean - ci, mean + ci

    ax.fill_between(plot_torch(test_x), plot_torch(y1), plot_torch(y2), color='g', alpha=.4, label='$3\sigma$')

    ax.set_title(title, fontsize=fs)
    ax.set_yticks(np.array((-200,0,200)))
    ax.set_ylim([-300,300])
    ax.tick_params(axis='both', which='major', labelsize=fs)

    if legend_true:
        ax.legend()