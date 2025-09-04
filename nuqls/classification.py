import torch
from torch.func import functional_call
from torch.utils.data import DataLoader
import tqdm
import copy
import nuqls.utils as utils
from torchmetrics.classification import MulticlassCalibrationError

torch.set_default_dtype(torch.float64)

class classificationParallel(object):
    def __init__(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device
        self.params = {k: v.detach() for k, v in self.network.named_parameters()}
        print(f'NUQLS is using device {self.device}.')

    def fnet(self, params, x):
        return functional_call(self.network, params, x)

    def jvp_first(self, theta_s,params,x):
        dparams = utils._dub(utils.unflatten_like(theta_s, params.values()),params)
        _, proj = torch.func.jvp(lambda param: self.fnet(param, x),
                                (params,), (dparams,))
        proj = proj.detach()
        return proj
    
    def vjp_second(self,resid_s,params,x):
        _, vjp_fn = torch.func.vjp(lambda param: self.fnet(param, x), params)
        vjp = vjp_fn(resid_s)
        return vjp

    def train(self, train, train_bs, n_output, scale, S, epochs, lr, mu, threshold = None, verbose=False, extra_verbose=False, save_weights=None):
        
        train_loader = DataLoader(train,train_bs)

        num_p = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        p = copy.deepcopy(self.params).values()
        theta_S = torch.empty((num_p,S), device=self.device)
        
        for s in range(S):
            theta_star = []
            for pi in p:
                theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=self.device)*scale + pi.to(self.device)))
            theta_S[:,s] = utils.flatten(theta_star).detach()

        theta_S = theta_S.detach()
        
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        if save_weights is not None:
            save_dict = {}
            save_dict['info'] = {
                                'gamma': scale,
                                'epochs': epochs,
                                'lr': lr,
                                }
            save_dict['training'] = {}
        
        with torch.no_grad():
            for epoch in pbar:
                loss = torch.zeros((S), device='cpu')
                acc = torch.zeros((S), device='cpu')

                if extra_verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,y in pbar_inner:
                    x, y = x.to(device=self.device, non_blocking=True, dtype=torch.float64), y.to(device=self.device, non_blocking=True)
                    f_nlin = self.network(x)
                    proj = torch.vmap(self.jvp_first, (1,None,None))((theta_S),self.params,x).permute(1,2,0)
                    f_lin = (proj + f_nlin.unsqueeze(2))
                    Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
                    ybar = torch.nn.functional.one_hot(y,num_classes=n_output)
                    resid = (Mubar - ybar.unsqueeze(2))
                    projT = torch.vmap(self.vjp_second, (2,None,None))(resid,self.params,x)
                    vjp = [j.detach().flatten(1) for j in projT[0].values()]
                    vjp = torch.cat(vjp,dim=1).detach()
                    g = (vjp.T / x.shape[0]).detach()

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    l = (-1 / x.shape[0] * torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1))).detach().cpu()
                    loss += l

                    # Early stopping
                    if threshold:
                        bt[:,l < threshold] = -bt[:,l < threshold]
                        bt[:,l == threshold] = 0

                    theta_S -= lr*bt

                    a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    acc += a

                    if extra_verbose:
                        ma_l = loss / (pbar_inner.format_dict['n'] + 1)
                        ma_a = acc / ((pbar_inner.format_dict['n'] + 1) * x.shape[0])
                        metrics = {'min_loss_ma': ma_l.min().item(),
                                'max_loss_batch': ma_l.max().item(),
                                'min_acc_batch': ma_a.min().item(),
                                'max_acc_batch': ma_a.max().item(),
                                    'resid_norm': torch.mean(torch.square(g)).item()}
                        if self.device.type == 'cuda':
                            metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                        else:
                            metrics['gpu_mem'] = 0
                        pbar_inner.set_postfix(metrics)

                loss /= len(train_loader)
                acc /= len(train)

                if verbose:
                    metrics = {'min_loss': loss.min().item(),
                               'max_loss': loss.max().item(),
                               'min_acc': acc.min().item(),
                               'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                    if self.device.type == 'cuda':
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    pbar.set_postfix(metrics)

                if save_weights is not None:
                    save_dict['training'][f'{epoch}'] = {
                                'weights': theta_S,
                                'min_loss': loss.min().item(),
                                'max_loss': loss.max().item(),
                                'min_acc': acc.min().item(),
                                'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()
                                }
                    torch.save(save_dict, save_weights)

        if verbose:
            print('Posterior samples computed!')
        self.theta_S = theta_S

        return loss.max().item(), acc.min().item()
    
    def pre_load(self, pre_load):
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta_S = weight_dict['training'][l]['weights']
        
    def test(self, test, test_bs=50):

        test_loader = DataLoader(test, test_bs)
        
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            f_nlin = self.network(x)
            proj = torch.vmap(self.jvp_first, (1,None,None))((self.theta_S),self.params,x).permute(1,2,0)
            f_lin = (proj + f_nlin.unsqueeze(2))
            pred_s.append(f_lin.detach())

        id_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C
        
        del f_lin
        del pred_s
    
        return id_predictions
    
    def UncertaintyPrediction(self, test, test_bs):

        logits = self.test(test, test_bs)

        probits = logits.softmax(dim=2)
        mean_prob = probits.mean(0)
        var_prob = probits.var(0)

        return mean_prob.detach(), var_prob.detach()
    
    def eval(self,x):
        x = x.to(self.device)
        f_nlin = self.network(x)
        proj = torch.vmap(self.jvp_first, (1,None,None))((self.theta_S),self.params,x).permute(1,2,0)
        f_lin = (proj + f_nlin.unsqueeze(2))
        return f_lin.detach().permute(2,0,1) 
    
    def HyperparameterTuning(self, validation, metric = 'ece', left = 1e-2, right = 1e2, its = 100, verbose=False):
        predictions = self.test(validation, test_bs=200)

        loader = DataLoader(validation,batch_size=100)
        val_targets = torch.cat([y.to(self.device) for _,y in loader])

        if metric == 'ece':
            ece_compute = MulticlassCalibrationError(num_classes=self.num_output,n_bins=10,norm='l1')
            def ece_eval(gamma):
                mean_prediction = (predictions * gamma).softmax(-1).mean(0)
                return ece_compute(mean_prediction, val_targets).cpu().item()
            f = ece_eval
            input_name, output_name = 'scale', 'ECE' 
        elif metric == 'varroc-id':
            def varroc_id_eval(gamma):
                scaled_predictions = (predictions * gamma).softmax(-1)
                idc, idic = utils.sort_preds(scaled_predictions,val_targets)
                return utils.aucroc(idic.var(0).sum(1), idc.var(0).sum(1))
            f = lambda x : -varroc_id_eval(x)
            input_name, output_name = 'scale', 'VARROC-ID' 
        else:
            print('Invalid metric choice. Valid choices are: [ece].')

        scale = utils.ternary_search(f = f,
                               left=left,
                               right=right,
                               its=its,
                               verbose=verbose,
                               input_name=input_name,
                               output_name=output_name)

        self.scale_cal = scale