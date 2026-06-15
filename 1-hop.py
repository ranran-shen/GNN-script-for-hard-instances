import torch
from SAGE_engine import train_sage   # the graph-agnostic engine
from SAGE_engine import drawGraph

def diagnose_degrees(deg, y_paper, max_bin=30):
    for label in [0, 1]:
        d = deg[y_paper == label]
        print(f"\n===== label {label}  (n={d.numel()}) =====")
        print(f"  mean   = {d.float().mean().item():.4f}")
        print(f"  std    = {d.float().std().item():.4f}")
        print(f"  min    = {d.min().item()}")
        print(f"  max    = {d.max().item()}")
        print(f"  median = {d.median().item()}")
        print(f"  #(deg==0) = {(d == 0).sum().item()}")   # should be 0 after truncate/shift

        print("  histogram:")
        counts = torch.bincount(d.clamp(max=max_bin), minlength=max_bin + 1)
        peak = counts.max().item()
        for v in range(max_bin + 1):
            c = counts[v].item()
            if c == 0:
                continue
            bar = "█" * int(50 * c / peak)
            tag = f"{v}" if v < max_bin else f"{v}+"
            print(f"    {tag:>4} | {bar} {c}")


# avoid isolated paper nodes
def build_1hop_hard_instance(num_papers=5000, feat_dim=32,
                             rate0=2.0, rate1=15.0, author_feat="gaussian", 
                             degree_mode="raw",
                             seed=0):
    g = torch.Generator().manual_seed(seed)

    # paper labels: first half -> 0, second half -> 1
    half = num_papers // 2
    y_paper = torch.cat([torch.zeros(half), torch.ones(num_papers - half)]).long()

    # number of authors per paper ~ Poisson(rate by label)
    rates = torch.full((num_papers,), rate0)
    rates[y_paper == 1] = rate1

    if degree_mode == "shift":
        deg = torch.poisson(rates, generator=g).long() + 1
    elif degree_mode == "truncate":
        deg = torch.poisson(rates, generator=g).long()
        # rejection sampling: 把 ==0 的位置反复重采，直到没有 0
        zero_mask = (deg == 0)
        while zero_mask.any():
            deg[zero_mask] = torch.poisson(rates[zero_mask], generator=g).long()
            zero_mask = (deg == 0)
    else:  # "raw"
        deg = torch.poisson(rates, generator=g).long()

        # Label 0: deg==0 count = 315 ratio = 0.126
        # Label 1: deg==0 count = 0 ratio = 0.0
    
    # diagnose_degrees(deg, y_paper)

    num_authors = int(deg.sum().item())
    N = num_papers + num_authors

    # each author writes exactly ONE paper -> star graph, no author sharing.
    # author ids start at num_papers and are handed out contiguously per paper.
    author_ids = num_papers + torch.arange(num_authors)
    paper_of_author = torch.repeat_interleave(torch.arange(num_papers), deg)
    src = torch.cat([paper_of_author, author_ids])       # paper -> author
    dst = torch.cat([author_ids, paper_of_author])       # author -> paper (both directions)
    edge_index = torch.stack([src, dst]).long()

    # features: random Gaussian for everyone (papers AND authors)
    x = torch.randn(N, feat_dim, generator=g)

    if author_feat == "zero":
        # authors to be 0
        x[num_papers:] = 0.0
        print("author's feature: 0!!!!!")

    # node labels: only papers carry the true label, authors get 0 placeholder
    y = torch.zeros(N, dtype=torch.long)
    y[:num_papers] = y_paper

    # stratified split of paper nodes into train / val / test
    train, val, test = [], [], []
    for c in (0, 1):
        ids = (y_paper == c).nonzero(as_tuple=True)[0]
        n = ids.numel()
        perm = ids[torch.randperm(n, generator=g)]
        k1, k2 = int(0.6 * n), int(0.8 * n)
        train.append(perm[:k1])
        val.append(perm[k1:k2])
        test.append(perm[k2:])

    return x, edge_index, y, torch.cat(train), torch.cat(val), torch.cat(test)


if __name__ == "__main__":

    feature_dim = 32
    dropout = 0.0

    fanout_list_some = [[5],
                   [5, 7],
                   [5, 7, 2]]

    # IMPORTANT: fanout=None  -> full neighborhood (keeps author degree).
    fanout_list_all = [[None],
                       [None, None],
                       [None, None, None]]
    

    af_list = ["gaussian"] # ["zero"] # ["gaussian", "zero"]
    simple_mode_list = [True, False] # simple_mode_list = [True, False]
    degree_mode_list = ["truncate"] # ["truncate", "shift", "raw"]
    sample_some_neighbor_mode = [False] # [True, False], False means that use all neighbor

    for af in af_list:
        for simple_mode in simple_mode_list:
            print(f"\n=== simple_mode={simple_mode!r}")
            for degree_mode in degree_mode_list:
                for sample_some_neighbor in sample_some_neighbor_mode:
                    if sample_some_neighbor:
                        fanout_list = fanout_list_some
                    else:
                        fanout_list = fanout_list_all

                    train_acc_L_list = []
                    val_acc_L_list = []
                    test_acc_L_list= []

                    for L in (1, 2, 3):
                        x, ei, y, tr, va, te = build_1hop_hard_instance(feat_dim=feature_dim,author_feat=af,degree_mode=degree_mode)

                        train_acc_list, val_acc_list, test_acc_list = train_sage(x, ei, y, tr, va, te, 
                                                                                num_layers=L, 
                                                                                fanout = fanout_list[L-1], 
                                                                                eval_fanout=None,
                                                                                dropout=dropout,
                                                                                device="cuda",
                                                                                simple_mode=simple_mode)
                        

                        train_acc_L_list.append(train_acc_list)
                        val_acc_L_list.append(val_acc_list)
                        test_acc_L_list.append(test_acc_list)



                    tag = ""
                    tag += af # gaussian
                    tag += "-"
                    tag += degree_mode # truncate
                    tag += "-some" if sample_some_neighbor else "-all"
                    tag += "-simple" if simple_mode else ""
                    save_path = f"../result/1-hop/{tag}.png"

                    
                    drawGraph(train_acc_L_list, val_acc_L_list, test_acc_L_list,
                            save_path=save_path)
