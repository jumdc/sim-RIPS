"""
RipsLayer class in PyTorch. [1]

References
----------
[1] https://gudhi.inria.fr/python/latest/rips_complex_tflow_itf_ref.html

"""


import torch
import numpy as np
from gudhi import RipsComplex

# The parameters of the model are the point coordinates.

@torch.compile
def Rips(distance_X, mel, dim, card):
    # Parameters: distance_X (distance matrix), 
    #             mel (maximum edge length for Rips filtration), 
    #             dim (homological dimension), 
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Compute the persistence pairs with Gudhi
    rc = RipsComplex(distance_matrix=distance_X, max_edge_length=mel)
    st = rc.create_simplex_tree(max_dimension=dim+1)
    dgm = st.persistence()
    pairs = st.persistence_pairs()

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for s1, s2 in pairs:
        if len(s1) == dim+1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(distance_X[l1,:][:,l1]),[len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(distance_X[l2,:][:,l2]),[len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(st.filtration(s2) - st.filtration(s1))
    
    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1,4])[perm][::-1,:].flatten())
    
    # Output indices
    indices = indices[:4*card] + [0 for _ in range(0,max(0,4*card-len(indices)))]
    return list(np.array(indices, dtype=np.int32))


class RipsModule(torch.nn.Module):
    """RipsLayer class in PyTorch. """
    def __init__(self, homology_dimensions, maximum_edge_length=np.inf, min_persistence=None, homology_coeff_field=11,*args, **kwargs):
        super().__init__()
        self.max_edge = maximum_edge_length
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0. for _ in range(len(self.dimensions))]
        self.hcf = homology_coeff_field
        assert len(self.min_persistence) == len(self.dimensions)

    def forward(self, X):
        """Forward pass of the layer. """
        # l2 distance
        d_X = torch.sqrt(torch.sum((torch.unsqueeze(X, 1)-torch.unsqueeze(X, 0))**2, 2))
        d_XX = torch.reshape(d_X, [1, d_X.shape[0], d_X.shape[1]])
        # Rips
        rips_torch = torch.compile(Rips)

        diag = Rips(d_XX, self.max_edge, self.dimensions, self.hcf)
        
        