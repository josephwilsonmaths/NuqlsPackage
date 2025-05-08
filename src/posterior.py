import torch
import src.regression
import src.regressionfull
import src.classification

def Nuqls(
    network: torch.nn.Module,
    task: str ='regression',
    full_dataset: bool = False
    ):
    
    if task == 'regression':
        if full_dataset:
            return src.regressionfull.regressionParallelFull(network)
        else:
            return src.regression.regressionParallel(network)
    
    elif task == 'classification':
        return src.classification.classificationParallel(network)
    
