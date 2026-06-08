import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


# draw the accuracy line
# not importtant
def drawGraph(train_acc_L_list, val_acc_L_list, test_acc_L_list, 
              save_path):
    epochs = np.arange(len(train_acc_L_list[0]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Train
    for i, L in enumerate([1, 2, 3]):
        axes[0].plot(
            epochs,
            train_acc_L_list[i],
            color=colors[i],
            label=f'L={L}'
        )

    axes[0].set_title("Train Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Validation
    for i, L in enumerate([1, 2, 3]):
        axes[1].plot(
            epochs,
            val_acc_L_list[i],
            color=colors[i],
            label=f'L={L}'
        )

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)


    # Test
    for i, L in enumerate([1, 2, 3]):
        axes[2].plot(
            epochs,
            test_acc_L_list[i],
            color=colors[i],
            label=f'L={L}'
        )

    axes[2].set_title("Test Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True)


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# get the adjacancy list of the input graph
def build_adj(edge_index, num_nodes):
    """edge_index (assumed symmetric) -> list of neighbor LongTensors."""
    adj = [[] for _ in range(num_nodes)]
    s, d = edge_index.tolist()
    for u, v in zip(s, d):
        adj[u].append(v)
    return [torch.tensor(a, dtype=torch.long) for a in adj]


# sample some neighbors
def khop_sample(seed_ids, adj, x, fanouts, g):
    """Generic BFS sampler over ANY graph.

    fanouts[h] neighbors are drawn at hop h+1; fanouts[h]=None -> take ALL
    neighbors (degree-preserving). 
    
    Returns (sub_x, edge_index, seed_local, n).
    """
    id2local, src, dst = {}, [], []

    def local(gid):
        if gid not in id2local:
            id2local[gid] = len(id2local)
        return id2local[gid]

    seed_local = [local(s) for s in seed_ids.tolist()]
    frontier = seed_ids.tolist()

    for f in fanouts:
        nxt = [] # store all the next layer nodes
        for u in frontier: # for every seed nodes
            # get neighbors
            lu = id2local[u] 
            nbrs = adj[u]

            if nbrs.numel() == 0: # have no neighbors
                continue

            # select f neighbors
            sel = nbrs if f is None else nbrs[torch.randint(0, nbrs.numel(), (f,), generator=g)]

            for v in sel.tolist():
                lv = local(v)
                src += [lu, lv]; dst += [lv, lu]
                nxt.append(v)
        frontier = nxt

    n = len(id2local)
    inv = torch.empty(n, dtype=torch.long)
    for gid, lid in id2local.items():
        inv[lid] = gid
    return (x[inv], torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(seed_local, dtype=torch.long), n)


# aggregation
def mean_aggregate(x, edge_index, num_nodes):
    # x: (num_nodes, dim)
    # (src_index, dst_index) = (edge_index[0][i], edge_index[1][i]) is an edge
    src, dst = edge_index

    out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
    out.index_add_(0, dst, x[src])

    deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, torch.ones(src.numel(), device=x.device, dtype=x.dtype))
    return out / deg.clamp(min=1).unsqueeze(1)



# one layer of GraphSAGE
class SAGEConvScratch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # parameters to be trained
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, num_nodes):
        agg = mean_aggregate(x, edge_index, num_nodes) # aggregation
        return self.lin_self(x) + self.lin_neigh(agg) 
    


class SAGE(nn.Module):
    def __init__(self, in_dim, hid, n_classes, num_layers, dropout=0.5, simple_mode=True):
        super().__init__()

        self.simple = simple_mode

        if self.simple: # simple SAGE
            self.convs = nn.ModuleList()
            if num_layers == 1:
                self.convs.append(SAGEConvScratch(in_dim, n_classes))
            else:
                self.convs.append(SAGEConvScratch(in_dim, hid))
                for _ in range(num_layers - 2):
                    self.convs.append(SAGEConvScratch(hid, hid))
                self.convs.append(SAGEConvScratch(hid, n_classes))
        else:
            self.convs = nn.ModuleList()
            d = in_dim
            for _ in range(num_layers):
                self.convs.append(SAGEConvScratch(d, hid)); d = hid
            
            # hid-dimension embedding -> class
            # a small network to predict
            self.readout = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hid, n_classes))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, seed_idx, num_nodes):
        if self.simple: # simple SAGE, with only ReLU between layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index, num_nodes)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
            return x[seed_idx]
        else: # more complex, with relu and dropout for every layer
            for conv in self.convs:
                x = F.dropout(F.relu(conv(x, edge_index, num_nodes)),
                            p=self.dropout, training=self.training)
            return self.readout(x[seed_idx]) # only predict seed idx, predict with a small network


def train_sage(x, edge_index, y, train_ids, val_ids, test_ids, num_layers,
               fanout=5, eval_fanout=None, n_classes=2, hid=32, steps=200, bs=128,
               lr=1e-2, weight_decay=5e-4, dropout=0.5, device="cpu", eval_every=20, simple_mode=True):
    """Train on ANY graph. fanout=None -> full-neighborhood (degree-preserving)."""

    # x : (N, feat_dim), N is number of nodes in the whole graph

    torch.manual_seed(0)
    x, y = x.to(device), y.to(device)
    train_ids, val_ids, test_ids = train_ids.to(device), val_ids.to(device), test_ids.to(device)
    
    # get the adjacency list of the whole graph
    adj = build_adj(edge_index, x.size(0))

    
    # deal with fanout
    if isinstance(fanout, int) or fanout is None:
        train_fanouts = [fanout] * num_layers
    else:
        train_fanouts = fanout
        assert len(train_fanouts) == num_layers
    
    if isinstance(eval_fanout, int) or eval_fanout is None:
        eval_fanouts = [eval_fanout] * num_layers
    else:
        eval_fanouts = eval_fanout
    

    # SAGE.__init__(in_dim, hid, n_classes, num_layers, dropout=0.5)
    model = SAGE(x.size(1), hid, n_classes, num_layers, dropout, simple_mode=simple_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gtr = torch.Generator().manual_seed(1)
    

    def evaluate(ids, gseed):
        # ids is val_ids or test_ids
        model.eval()
        ge = torch.Generator().manual_seed(gseed)
        with torch.no_grad():
            sx, ei, sl, n = khop_sample(ids.cpu(), adj, x, eval_fanouts, ge)

            pred = model(sx.to(device), ei.to(device), sl.to(device), n).argmax(1) # x, edge_index, seed_idx, num_nodes
            return (pred == y[ids]).float().mean().item()


    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    for step in range(steps):
        model.train()
        opt.zero_grad()

        # get batch_size = bs paper nodes from train set
        batch = train_ids[torch.randperm(train_ids.numel(), generator=gtr)[:bs]].to(device)

        sx, ei, sl, n = khop_sample(batch.cpu(), adj, x, train_fanouts, gtr)
        loss = F.cross_entropy(model(sx.to(device), ei.to(device), sl.to(device), n),
                               y[batch])
        loss.backward()
        opt.step()

        with torch.no_grad():
            batch_logits = model(sx.to(device), ei.to(device), sl.to(device), n)
            train_batch_acc = (batch_logits.argmax(1) == y[batch]).float().mean().item()
            train_acc_list.append(train_batch_acc)
        
        
        # calculate the accuracy on validation set and test set
        va = evaluate(val_ids, gseed=7)
        te = evaluate(test_ids, gseed=8)
        val_acc_list.append(va)
        test_acc_list.append(te)
        
        
        if step % eval_every == 0:
            print(f"step: {step}")

    return train_acc_list, val_acc_list, test_acc_list
