import torch
from torch.func import functional_call
from torch.utils.data import DataLoader
import tqdm
import copy
import nuqls.utils as utils
from torchmetrics.classification import MulticlassCalibrationError
from functorch import make_functional
from torch.func import vmap, jacrev

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

    def train(self, train, train_bs, n_output, scale, S, epochs, lr, mu, wd = 0, threshold = None, verbose=False, extra_verbose=False, save_weights=None):
        
        train_loader = DataLoader(train,train_bs)

        if train_bs >= len(train):
            X,_ = next(iter(train_loader))
            fnet, paramsf = make_functional(self.network)
            self.theta_t = utils.flatten(paramsf)
            # Compute jacobian of net, evaluated on training set
            def fnet_single(params, x):
                return fnet(params, x.unsqueeze(0)).squeeze(0)
            J = vmap(jacrev(fnet_single), (None, 0))(paramsf, X.to(self.device))
            J = [j.detach().flatten(2) for j in J]
            J = torch.cat(J,dim=2).detach()
        else:
            J = None

        num_p = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        p = copy.deepcopy(self.params).values()
        theta_S = torch.empty((num_p,S), device=self.device)
        
        for s in range(S):
            theta_star = []
            for pi in p:
                theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=self.device)*scale + pi.to(self.device)))
            theta_S[:,s] = utils.flatten(theta_star).detach()

        theta_S = theta_S.detach()
        self.theta_init = theta_S.detach().clone()
        
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
                    if J is not None:
                        f_lin = ((J @ (theta_S.to(self.device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                                f_nlin.reshape(-1,1)).reshape(x.shape[0],n_output,S)
                    else:
                        proj = torch.vmap(self.jvp_first, (1,None,None))((theta_S),self.params,x).permute(1,2,0)
                        f_lin = (proj + f_nlin.unsqueeze(2))
                    if n_output > 1:
                        Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
                        ybar = torch.nn.functional.one_hot(y,num_classes=n_output)
                    else:
                        Mubar = torch.clamp(torch.nn.functional.sigmoid(f_lin),1e-32,1)
                        ybar = y.unsqueeze(1)
                    resid = (Mubar - ybar.unsqueeze(2))
                    if J is not None:
                        g = torch.einsum('ncp,ncs->ps',J,resid) / x.shape[0]
                    else:
                        projT = torch.vmap(self.vjp_second, (2,None,None))(resid,self.params,x)
                        vjp = [j.detach().flatten(1) for j in projT[0].values()]
                        vjp = torch.cat(vjp,dim=1).detach()
                        g = (vjp.T / x.shape[0]).detach()

                    if wd > 0:
                        g += wd*theta_S

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    if n_output > 1:
                        l = (-1 / x.shape[0] * torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1))).detach().cpu()
                    else:
                        input1 = torch.clamp(-f_lin,max = 0) + torch.log(1 + torch.exp(f_lin.abs()))
                        input2 = -f_lin - input1
                        l = - torch.sum((ybar.unsqueeze(2) * (-input1) + (1 - ybar).unsqueeze(2) * input2),dim=(0,1)) / x.shape[0]
                    loss += l

                    # Early stopping
                    if threshold:
                        bt[:,l < threshold] *= -1
                        bt[:,l == threshold] = 0

                    theta_S -= lr*bt

                    if n_output > 1:
                        a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    else:
                        a = (f_lin.sigmoid().round().squeeze(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
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
        
    def test(self, test, test_bs=50, network_mean=False):

        test_loader = DataLoader(test, test_bs)
        
        # Concatenate predictions
        preds = []
        if network_mean:
            net_preds = []
        for x,_ in test_loader:
            preds.append(self.eval(x)) # S x N x C
            if network_mean:
                net_preds.append(self.network(x.to(self.device)))
        predictions = torch.cat(preds,dim=1) # N x C x S ---> S x N x C
        if network_mean:
            predictions_net = torch.cat(net_preds,dim=0)    
            return predictions, predictions_net
        else:
            return predictions
    
    def UncertaintyPrediction(self, test, test_bs, network_mean=False):
        predictions = self.test(test, test_bs, network_mean)

        if not network_mean:
            probits = predictions.softmax(dim=-1)
            mean_prob = probits.mean(0)
            var_prob = probits.var(0)
        else:
            nuql_predictions, net_predictions = predictions
            mean_prob = net_predictions.softmax(dim=-1)
            var_prob = nuql_predictions.softmax(dim=-1).var(0)

        return mean_prob.detach().cpu(), var_prob.detach().cpu()
    
    def eval(self,x):
        x = x.to(self.device)
        f_nlin = self.network(x)
        proj = torch.vmap(self.jvp_first, (1,None,None))((self.theta_S),self.params,x).permute(1,2,0)
        f_lin = (proj + f_nlin.unsqueeze(2))
        return f_lin.detach().permute(2,0,1) 
    
    def HyperparameterTuning(self, validation, metric = 'ece', left = 1e-2, right = 1e2, its = 100, verbose=False):
        predictions = self.test(validation, test_bs=200).cpu()

        loader = DataLoader(validation,batch_size=100)
        val_targets = torch.cat([y for _,y in loader])

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

class classificationParallelInterpolation(object):
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

    def train(self, train, train_bs, n_output, scale, S, epochs, lr, mu, wd = 0, threshold = None, verbose=False, extra_verbose=False, save_weights=None):
        
        train_loader = DataLoader(train,train_bs)

        if train_bs >= len(train):
            X,_ = next(iter(train_loader))
            fnet, paramsf = make_functional(self.network)
            self.theta_t = utils.flatten(paramsf)
            def fnet_single(params, x):
                return fnet(params, x.unsqueeze(0)).squeeze(0)
            J = vmap(jacrev(fnet_single), (None, 0))(paramsf, X.to(self.device))
            J = [j.detach().flatten(2) for j in J]
            J = torch.cat(J,dim=2).detach()
        else:
            J = None

        num_p = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        p = copy.deepcopy(self.params).values()
        theta_S = torch.empty((num_p,S), device=self.device)
        
        for s in range(S):
            theta_star = []
            for pi in p:
                theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=self.device)*scale + pi.to(self.device)))
            theta_S[:,s] = utils.flatten(theta_star).detach()

        theta_S = theta_S.detach()
        self.theta_init = theta_S.detach().clone()
        
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
                loss = 0

                if extra_verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,_ in pbar_inner:
                    x = x.to(device=self.device, non_blocking=True, dtype=torch.float64)
                    if J is not None:
                        f = J.flatten(0,1) @ (theta_S.to(self.device) - self.theta_t.unsqueeze(1))
                        g = J.flatten(0,1).T @ f / (X.shape[0] * n_output)
                    else:
                        f = torch.vmap(self.jvp_first, (1,None,None))((theta_S),self.params,x).permute(1,2,0) # N x C x S
                        g = torch.vmap(self.vjp_second, (2,None,None))(f,self.params,x)
                        vjp = [j.detach().flatten(1) for j in g[0].values()]
                        vjp = torch.cat(vjp,dim=1).detach()
                        g = (vjp.T / (x.shape[0] * n_output)).detach()

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    l = (torch.square(f).sum()  / (x.shape[0] * n_output * S)).detach()
                    loss += l
                    theta_S -= lr*bt

                    if extra_verbose:
                        metrics = {'loss': l.item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                        if self.device.type == 'cuda':
                            metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                        else:
                            metrics['gpu_mem'] = 0
                        pbar_inner.set_postfix(metrics)
                
                loss /= len(train_loader)

                if verbose:
                    metrics = {'loss': loss.item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                    if self.device.type == 'cuda':
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    pbar.set_postfix(metrics)

                if save_weights is not None:
                    save_dict['training'][f'{epoch}'] = {
                                'weights': theta_S,
                                'loss': loss.item(),
                                'resid_norm': torch.mean(torch.square(g)).item()
                                }
                    torch.save(save_dict, save_weights)

        if verbose:
            print('Posterior samples computed!')
        self.theta_S = theta_S

        return loss.item()
    
    def pre_load(self, pre_load):
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta_S = weight_dict['training'][l]['weights']
        
    def test(self, test, test_bs=50, network_mean=False):

        test_loader = DataLoader(test, test_bs)
        
        # Concatenate predictions
        preds = []
        if network_mean:
            net_preds = []
        for x,_ in test_loader:
            preds.append(self.eval(x)) # S x N x C
            if network_mean:
                net_preds.append(self.network(x.to(self.device)))
        predictions = torch.cat(preds,dim=1) # N x C x S ---> S x N x C
        if network_mean:
            predictions_net = torch.cat(net_preds,dim=0)    
            return predictions, predictions_net
        else:
            return predictions
    
    def UncertaintyPrediction(self, test, test_bs, network_mean=False):
        predictions = self.test(test, test_bs, network_mean)

        if not network_mean:
            probits = predictions.softmax(dim=-1)
            mean_prob = probits.mean(0)
            var_prob = probits.var(0)
        else:
            nuql_predictions, net_predictions = predictions
            mean_prob = net_predictions.softmax(dim=-1)
            var_prob = nuql_predictions.softmax(dim=-1).var(0)

        return mean_prob.detach().cpu(), var_prob.detach().cpu()
    
    def eval(self,x):
        x = x.to(self.device)
        f_nlin = self.network(x)
        proj = torch.vmap(self.jvp_first, (1,None,None))((self.theta_S),self.params,x).permute(1,2,0)
        f_lin = (proj + f_nlin.unsqueeze(2))
        return f_lin.detach().permute(2,0,1) 
    
    def HyperparameterTuning(self, validation, metric = 'ece', left = 1e-2, right = 1e2, its = 100, verbose=False):
        predictions = self.test(validation, test_bs=200).cpu()

        loader = DataLoader(validation,batch_size=100)
        val_targets = torch.cat([y for _,y in loader])

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