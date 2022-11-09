import torch

def get_distance(pos,edge_index,offset,cell,batch):
    pos_i=pos[edge_index[0]]
    pos_j=pos[edge_index[1]]
    edge_vec=pos_j-pos_i
    if cell.shape[0]>1:
        edge_vec=edge_vec+torch.einsum("ni,nij->nj", offset, cell[batch[edge_index[0]]])
    else:
        edge_vec=edge_vec+torch.einsum("ni,ij->nj",offset,cell.squeeze(0))
    length=torch.linalg.norm(edge_vec, dim=-1)

    return length,edge_vec