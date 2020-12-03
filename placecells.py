# %%
import numpy as np
import nengo


#%%
pos = np.array([0,0])
x_placecells = 3
y_placecells = 3
n_pc = x_placecells * y_placecells
diameter = 2
centers_x = centers_y = np.linspace(-diameter/2, diameter/2, x_placecells)
X,Y = np.meshgrid(centers_x, centers_y)
centers = np.array([X.flatten(), Y.flatten()]).T
print(centers.shape)


#%%

def distances(pos, sigma, centers):
    '''
    compute activity of all place cells for a given position,
    firing rate for each place cell is given by a gaussian centered around it's preferred
    location with width sigma
    returns a vector of activation of all place cells

    INPUTS:
        pos         -   position to be encoded
        sigma       -   width of gaussian
        centers     -   means of gaussians, shape [N x 2]

    RETURNS:
        activations -   vector encoding activities of all place cells
    '''

    distances = np.sqrt(np.sum((pos - centers)**2,axis=1)) # euclidean distances
    return distances

dists = distances(pos, 1, centers)

#%%
dists.shape

with nengo.Network() as model:
    dists = np.array([nengo.dists.Gaussian(0,1) for _ in range(n_pc)])
    dists = dists[None,:]
    nengo.Ensemble(n_neurons=1, dimensions=n_pc, encoders=dists)
