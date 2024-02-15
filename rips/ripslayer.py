"""
RipsLayer class in PyTorch. [1]

# References
# ----------
# [1] https://gudhi.inria.fr/python/latest/rips_complex_tflow_itf_ref.html

# """


# import torch
# import numpy as np
# from gudhi import RipsComplex

# # The parameters of the model are the point coordinates.

# def Rips(distance_X, mel, dim, card):
#     # Parameters: distance_X (distance matrix), 
#     #             mel (maximum edge length for Rips filtration), 
#     #             dim (homological dimension), 
#     #             card (number of persistence diagram points, sorted by distance-to-diagonal)

#     # Compute the persistence pairs with Gudhi
#     rc = RipsComplex(distance_matrix=distance_X, max_edge_length=mel)
#     st = rc.create_simplex_tree(max_dimension=dim+1)
#     dgm = st.persistence()
#     pairs = st.persistence_pairs()

#     # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
#     # distance among all pairwise distances between the simplex vertices
#     indices, pers = [], []
#     for s1, s2 in pairs:
#         if len(s1) == dim+1 and len(s2) > 0:
#             l1, l2 = np.array(s1), np.array(s2)
#             i1 = [s1[v] for v in np.unravel_index(np.argmax(distance_X[l1,:][:,l1]),[len(s1), len(s1)])]
#             i2 = [s2[v] for v in np.unravel_index(np.argmax(distance_X[l2,:][:,l2]),[len(s2), len(s2)])]
#             indices += i1
#             indices += i2
#             pers.append(st.filtration(s2) - st.filtration(s1))
    
#     # Sort points with distance-to-diagonal
#     perm = np.argsort(pers)
#     indices = list(np.reshape(indices, [-1,4])[perm][::-1,:].flatten())
    
#     # Output indices
#     indices = indices[:4*card] + [0 for _ in range(0,max(0,4*card-len(indices)))]
#     return list(np.array(indices, dtype=np.int32))


# class RipsModule(torch.nn.Module):
#     """RipsLayer class in PyTorch. """
#     def __init__(self, homology_dimensions, maximum_edge_length=np.inf, min_persistence=None, homology_coeff_field=11,*args, **kwargs):
#         super().__init__()
#         self.max_edge = maximum_edge_length
#         self.dimensions = homology_dimensions
#         self.min_persistence = min_persistence if min_persistence is not None else [0. for _ in range(len(self.dimensions))]
#         self.hcf = homology_coeff_field
#         assert len(self.min_persistence) == len(self.dimensions)

#     def forward(self, X):
#         """Forward pass of the layer. """
#         # l2 distance
#         d_X = torch.sqrt(torch.sum((torch.unsqueeze(X, 1)-torch.unsqueeze(X, 0))**2, 2))
#         # Rips
#         indices = Rips(d_XX, self.max_edge, self.dimensions, self.hcf)

#         # diag = Rips(d_XX, self.max_edge, self.dimensions, self.hcf)
#         self.dgms = []
#         for idx_dim, dimension in enumerate(self.dimensions):
#             cur_idx = indices[idx_dim]
#             if dimension > 0:
#                 finite_dgm = tf.reshape(tf.gather_nd(d_X, tf.reshape(cur_idx[0], [-1,2])), [-1,2])
#                 essential_dgm = tf.reshape(tf.gather_nd(d_X, tf.reshape(cur_idx[1], [-1,2])), [-1,1])
#             # else:
#             #     reshaped_cur_idx = tf.reshape(cur_idx[0], [-1,3])
#             #     finite_dgm = tf.concat([tf.zeros([reshaped_cur_idx.shape[0],1]), tf.reshape(tf.gather_nd(d_X, reshaped_cur_idx[:,1:]), [-1,1])], axis=1)
#             #     essential_dgm = tf.zeros([cur_idx[1].shape[0],1])
#             # min_pers = self.min_persistence[idx_dim]
#             # if min_pers >= 0:
#             #     persistent_indices = tf.where(tf.math.abs(finite_dgm[:,1]-finite_dgm[:,0]) > min_pers)
#             #     self.dgms.append((tf.reshape(tf.gather(finite_dgm, indices=persistent_indices),[-1,2]), essential_dgm))
#             # else:
#             #     self.dgms.append((finite_dgm, essential_dgm))
#         return self.dgms        
        

import numpy               as np
import tensorflow          as tf
# from ..rips_complex     import RipsComplex
from gudhi import RipsComplex

############################
# Vietoris-Rips filtration #
############################

# The parameters of the model are the point coordinates.

def _Rips(DX, max_edge, dimensions, homology_coeff_field):
    # Parameters: DX (distance matrix), 
    #             max_edge (maximum edge length for Rips filtration), 
    #             dimensions (homology dimensions)

    # Compute the persistence pairs with Gudhi
    rc = RipsComplex(distance_matrix=DX, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=max(dimensions)+1)
    st.compute_persistence(homology_coeff_field=homology_coeff_field)
    print(st)
    pairs = st.flag_persistence_generators()

    L_indices = []
    for dimension in dimensions:

        if dimension == 0:
            finite_pairs = pairs[0]
            essential_pairs = pairs[2]
        else:
            finite_pairs = pairs[1][dimension-1] if len(pairs[1]) >= dimension else np.empty(shape=[0,4])
            essential_pairs = pairs[3][dimension-1] if len(pairs[3]) >= dimension else np.empty(shape=[0,2])
        
        finite_indices = np.array(finite_pairs.flatten(), dtype=np.int32)
        essential_indices = np.array(essential_pairs.flatten(), dtype=np.int32)

        L_indices.append((finite_indices, essential_indices))

    return L_indices

class RipsLayer(tf.keras.layers.Layer):
    """
    TensorFlow layer for computing Rips persistence out of a point cloud
    """
    def __init__(self, homology_dimensions, maximum_edge_length=np.inf, min_persistence=None, homology_coeff_field=11, **kwargs):
        """
        Constructor for the RipsLayer class

        Parameters:
            maximum_edge_length (float): maximum edge length for the Rips complex 
            homology_dimensions (List[int]): list of homology dimensions
            min_persistence (List[float]): minimum distance-to-diagonal of the points in the output persistence diagrams (default None, in which case 0. is used for all dimensions)
            homology_coeff_field (int): homology field coefficient. Must be a prime number. Default value is 11. Max is 46337.
        """
        super().__init__(dynamic=True, **kwargs)
        self.max_edge = maximum_edge_length
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0. for _ in range(len(self.dimensions))]
        self.hcf = homology_coeff_field
        assert len(self.min_persistence) == len(self.dimensions)

        
    def call(self, X):
        """
        Compute Rips persistence diagram associated to a point cloud

        Parameters:   
            X (TensorFlow variable): point cloud of shape [number of points, number of dimensions]

        Returns:
            List[Tuple[tf.Tensor,tf.Tensor]]: List of Rips persistence diagrams. The length of this list is the same than that of dimensions, i.e., there is one persistence diagram per homology dimension provided in the input list dimensions. Moreover, the finite and essential parts of the persistence diagrams are provided separately: each element of this list is a tuple of size two that contains the finite and essential parts of the corresponding persistence diagram, of shapes [num_finite_points, 2] and [num_essential_points, 1] respectively
        """    
        # Compute distance matrix
        DX = tf.norm(tf.expand_dims(X, 1)-tf.expand_dims(X, 0), axis=2)
        # Compute vertices associated to positive and negative simplices 
        # Don't compute gradient for this operation
        indices = _Rips(DX.numpy(), self.max_edge, self.dimensions, self.hcf)
        # Get persistence diagrams by simply picking the corresponding entries in the distance matrix
        self.dgms = []
        for idx_dim, dimension in enumerate(self.dimensions):
            cur_idx = indices[idx_dim]
            if dimension > 0:
                finite_dgm = tf.reshape(tf.gather_nd(DX, tf.reshape(cur_idx[0], [-1,2])), [-1,2])
                essential_dgm = tf.reshape(tf.gather_nd(DX, tf.reshape(cur_idx[1], [-1,2])), [-1,1])
            else:
                reshaped_cur_idx = tf.reshape(cur_idx[0], [-1,3])
                finite_dgm = tf.concat([tf.zeros([reshaped_cur_idx.shape[0],1]), tf.reshape(tf.gather_nd(DX, reshaped_cur_idx[:,1:]), [-1,1])], axis=1)
                essential_dgm = tf.zeros([cur_idx[1].shape[0],1])
            min_pers = self.min_persistence[idx_dim]
            if min_pers >= 0:
                persistent_indices = tf.where(tf.math.abs(finite_dgm[:,1]-finite_dgm[:,0]) > min_pers)
                self.dgms.append((tf.reshape(tf.gather(finite_dgm, indices=persistent_indices),[-1,2]), essential_dgm))
            else:
                self.dgms.append((finite_dgm, essential_dgm))
        return self.dgms