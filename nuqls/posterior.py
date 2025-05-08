import torch
import nuqls.regression
import nuqls.regressionfull
import nuqls.classification

def Nuqls(
    network: torch.nn.Module,
    task: str ='regression',
    full_dataset: bool = False
    ):
    
    if task == 'regression':
        if full_dataset:
            return nuqls.regressionfull.regressionParallelFull(network)
        else:
            return nuqls.regression.regressionParallel(network)
    
    elif task == 'classification':
        return nuqls.classification.classificationParallel(network)
    
