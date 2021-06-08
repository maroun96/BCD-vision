import torch
import torch.optim as optim
import numpy as np 

class CD(optim.Optimizer):
    """ Randomized Coordinate Descent Algorithm
    """

    def __init__(self,params,lr,n_block, nb_params):
        defaults = dict(lr=lr,n_block = n_block, nb_params=nb_params)
        super(CD, self).__init__(params,defaults)

    @torch.no_grad()
    def step(self, closure = None ):
        
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            n_block = group["n_block"]
            nb_params = group["nb_params"]

            indexes = np.arange(nb_params)
            np.random.shuffle(indexes)
            rand_coord = indexes[:n_block]

            index = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                pv, d_pv = p.view(-1), d_p.view(-1)

                prev_index = index
                index += pv.size(0)
                select_coord = rand_coord[rand_coord < index]
                select_coord = select_coord[select_coord >= prev_index] - prev_index
                lr_tab = torch.zeros(index-prev_index)

                lr_tab[select_coord] = lr
                #print(lr_tab)
                pv.add_(-lr_tab*d_pv)

                #p.view(-1).add_(d_p.view(-1), alpha=-group['lr'])

        return loss