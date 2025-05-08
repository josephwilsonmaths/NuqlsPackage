import torch
from functorch import make_functional
from torch.utils.data import DataLoader
from torch.func import vmap, jacrev
import tqdm
import copy
import nuqls.utils as utils

torch.set_default_dtype(torch.float64)


class regressionParallel(object):
    def __init__(self, network):
        self.network = network
        self.device = next(network.parameters()).device

    def train(self, train, train_bs=50, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9, verbose=False):
        if train_bs >= len(train):
            return self.gd(
                train=train,
                scale=scale,
                S=S,
                epochs=epochs,
                lr=lr,
                mu=mu,
                verbose=verbose
            )
        else:
            return self.sgd(
                train=train,
                train_bs=train_bs,
                scale=scale,
                S=S,
                epochs=epochs,
                lr=lr,
                mu=mu,
                verbose=verbose
            )

    def gd(self, train, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9, verbose=False):
        fnet, params = make_functional(self.network)
        theta_t = utils.flatten(params)
        theta = torch.randn(size=(theta_t.shape[0],S),device=self.device)*scale + theta_t.unsqueeze(1)

        ## Create loaders and get entire training set
        train_loader_total = DataLoader(train,batch_size=len(train))
        X,Y = next(iter(train_loader_total))

        ## Compute jacobian of net, evaluated on training set
        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)
        J = vmap(jacrev(fnet_single), (None, 0))(params, X.to(self.device))
        J = [j.detach().flatten(1) for j in J]
        J = torch.cat(J,dim=1).detach()

        # Set progress bar
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)
        
        ## Train S realisations of linearised networks
        for epoch in pbar:
            if self.device == torch.device('cuda'):
                torch.cuda.reset_peak_memory_stats()

            X, Y = X.to(self.device), Y.to(self.device).reshape(-1,1)
            
            f_nlin = self.network(X)
            f_lin = (J.to(self.device) @ (theta - theta_t.unsqueeze(1)) + f_nlin).detach()
            resid = f_lin - Y
            grad = J.T.to(self.device) @ resid.to(self.device) / X.shape[0]

            if epoch == 0:
                bt = grad
            else:
                bt = mu*bt + grad

            theta -= lr * bt

            loss = torch.mean(torch.square(J @ (theta - theta_t.unsqueeze(1)) + f_nlin - Y)).max()

            if verbose:
                metrics = {'mean_loss': loss.item(),
                   'resid_norm': torch.mean(torch.square(grad)).item()}
                if self.device == torch.device('cuda'):
                    metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                else:
                    metrics['gpu_mem'] = 0
                pbar.set_postfix(metrics)

        # Report maximum loss over S, and the mean gradient norm
        max_l2_loss = torch.mean(torch.square(J @ (theta - theta_t.unsqueeze(1)) + f_nlin - Y)).item()
        norm_resid = torch.mean(torch.square(J.T @ ( f_nlin + J @ (theta - theta_t.unsqueeze(1)) - Y))).item()

        if verbose:
            print('Posterior samples computed!')
        self.theta = theta

        return max_l2_loss, norm_resid

    def sgd(self, train, train_bs=50, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9, verbose=False):
        fnet, params = make_functional(self.network)

        num_p = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        train_loader = DataLoader(train,train_bs)

        p = copy.deepcopy(params)
        theta_S = torch.empty((num_p,S), device=self.device)

        for s in range(S):
            theta_star = []
            for pi in p:
                theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=self.device)*scale + pi.to(self.device)))
            theta_S[:,s] = utils.flatten(theta_star)

        def jvp_first(theta_s,params,x):
            dparams = utils._sub(tuple(utils.unflatten_like(theta_s, params)),params)
            _, proj = torch.func.jvp(lambda param: fnet(param, x),
                                    (params,), (dparams,))
            return proj

        def vjp_second(resid_s,params,x):
            _, vjp_fn = torch.func.vjp(lambda param: fnet(param, x), params)
            vjp = vjp_fn(resid_s.unsqueeze(1))
            return vjp

        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        for epoch in pbar:
            loss = 0
            resid_norm = 0

            if self.device == torch.device('cuda'):
                torch.cuda.reset_peak_memory_stats()

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                f_nlin = self.network(x)
                proj = torch.vmap(jvp_first, (1,None,None))(theta_S,params,x).flatten(1).T
                f_lin = (proj + f_nlin).detach()
                resid = f_lin - y
                projT = torch.vmap(vjp_second, (1,None,None))(resid,params,x)
                vjp = [j.detach().flatten(1) for j in projT[0]]
                vjp = torch.cat(vjp,dim=1).detach()
                g = vjp.T / x.shape[0]

                if epoch == 0:
                    bt = g
                else:
                    bt = mu*bt + g

                theta_S -= lr*bt

                loss += torch.mean(torch.square(resid)).max().item()
                resid_norm += torch.mean(torch.square(g)).item()
            
            loss /= len(train_loader)
            resid_norm /= len(train_loader)

            if verbose:
                metrics = {'max_loss': loss,
                   'resid_norm': resid_norm}
                if self.device == torch.device('cuda'):
                    metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                else:
                    metrics['gpu_mem'] = 0
                pbar.set_postfix(metrics)

        if verbose:
            print('Posterior samples computed!')
        self.theta_S = theta_S

        return loss
        
    def test(self, test, test_bs=50):
        fnet, params = make_functional(self.network)

        test_loader = DataLoader(test, test_bs)

        def jvp_first(theta_s,params,x):
            dparams = utils._sub(tuple(utils.unflatten_like(theta_s, params)),params)
            _, proj = torch.func.jvp(lambda param: fnet(param, x),
                                    (params,), (dparams,))
            return proj

        # Concatenate predictions
        pred_test = []
        for x,_ in test_loader:
            x = x.to(self.device)
            f_nlin = self.network(x)
            proj = torch.vmap(jvp_first, (1,None,None))(self.theta_S,params,x).flatten(1).T
            pred = proj + f_nlin
            pred_test.append(pred.detach())

        id_predictions = torch.cat(pred_test,dim=0)
        
        return id_predictions
    
    def HyperparameterTuning(self, validation, left, right, its, verbose=False):
        val_predictions = self.test(validation)

        calibration_test_loader_val = DataLoader(validation,len(validation))
        _, val_y = next(iter(calibration_test_loader_val))

        left_scale, right_scale = left, right

        # Ternary Search
        for k in range(its):
            left_third = left_scale + (right_scale - left_scale) / 3
            right_third = right_scale - (right_scale - left_scale) / 3

            # Left ECE
            scaled_nuqls_predictions = val_predictions * left_third
            obs_map, predicted = utils.calibration_curve_r(val_y,val_predictions.mean(1),scaled_nuqls_predictions.var(1),11)
            left_ece = torch.mean(torch.square(obs_map - predicted))

            # Right ECE
            scaled_nuqls_predictions = val_predictions * right_third
            obs_map, predicted = utils.calibration_curve_r(val_y,val_predictions.mean(1),scaled_nuqls_predictions.var(1),11)
            right_ece = torch.mean(torch.square(obs_map - predicted))

            if left_ece > right_ece:
                left_scale = left_third
            else:
                right_scale = right_third

            scale = (left_scale + right_scale) / 2

            # Print info
            if verbose:
                print(f'\nSCALE {scale:.3}:: MAP: [{left_ece:.1%},{right_ece:.1%}]') 

            if abs(right_scale - left_scale) <= 1e-2:
                print('Converged.')
                break

        self.scale_cal = scale

    def CalibratedUncertaintyPrediction(self, test):

        test_predictions = self.test(test)
        
        # Scale test predictions
        nuqls_predictions = test_predictions*self.scale_cal

        mean_pred = test_predictions.mean(1)
        var_pred = nuqls_predictions.var(1) # We only find the gamma term in the variance

        return mean_pred, var_pred