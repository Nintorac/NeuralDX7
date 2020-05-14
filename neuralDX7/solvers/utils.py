import torch


def sigmoidal_annealing(iter_nb, t=1e-4, s=-6):
    """

    iter_nb - number of parameter updates completed
    t - step size
    s - slope of the sigmoid
    """
    
    t, s = torch.tensor(t), torch.tensor(s).float()
    x0 = torch.sigmoid(s)
    value = (torch.sigmoid(iter_nb*t + s) - x0)/(1-x0) 

    return value