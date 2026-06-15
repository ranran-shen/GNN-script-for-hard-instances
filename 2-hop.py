import torch
from SAGE_engine import train_sage   # the graph-agnostic engine
from SAGE_engine import drawGraph


def build_2hop_hard_instance(num_papers=5000, num_prolific=100, num_nonprolific=1900,
                             deg=8, feat_dim=32, author_feat="gaussian",
                             seed=0):
    
    g = torch.Generator().manual_seed(seed)

    num_authors = num_prolific + num_nonprolific
    N = num_papers + num_authors

    PROL0 = num_papers                       # prolific author ids start here
    NONP0 = num_papers + num_prolific        # non-prolific author ids start here

    # Node id layout:  papers [0, num_papers) | prolific | non-prolific
    half = num_papers // 2
    y_paper = torch.cat([torch.zeros(half), torch.ones(num_papers - half)]).long() # label 0 and 1
    prolific = torch.arange(PROL0, PROL0 + num_prolific)
    nonprolific = torch.arange(NONP0, NONP0 + num_nonprolific)

    # edges: each paper -> `deg` DISTINCT authors from its class pool (both directions)
    src, dst = [], []
    for p in range(num_papers):
        pool = prolific if y_paper[p].item() == 1 else nonprolific
        sel = pool[torch.randperm(pool.numel(), generator=g)[:deg]]
        for a in sel.tolist():
            src += [p, a]; dst += [a, p]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # features: random Gaussian for everyone (frozen)
    x = torch.randn(N, feat_dim, generator=g)
    if author_feat == "zero":
        x[num_papers:] = 0.0
        print("author's feature: 0!!!!!")

    # set label for paper node
    y = torch.zeros(N, dtype=torch.long)
    y[:num_papers] = y_paper





    # prolific_feats = x[PROL0 : PROL0 + num_prolific]       # shape (100, 32)
    # nonprolific_feats = x[NONP0 : NONP0 + num_nonprolific]  # shape (1900, 32)

    # mu_prolific = prolific_feats.mean(dim=0)      # shape (32,)
    # mu_nonprolific = nonprolific_feats.mean(dim=0) # shape (32,)

    # print("prolific centroid norm:    ", mu_prolific.norm().item())
    # print("nonprolific centroid norm: ", mu_nonprolific.norm().item())
    # print("centroid diff norm:        ", (mu_prolific - mu_nonprolific).norm().item())

    # prolific centroid norm:     0.487601101398468
    # nonprolific centroid norm:  0.11741432547569275
    # centroid diff norm:         0.48970070481300354



    # split paper nodes into: train, validation, test
    train, val, test = [], [], []
    for c in (0, 1):
        ids = (y_paper == c).nonzero(as_tuple=True)[0]
        n = ids.numel() # number of label-c paper

        perm = ids[torch.randperm(n, generator=g)] # perm = shuffled ids
        
        k1 = int(0.6 * n)
        k2 = int(0.8 * n)

        train.append(perm[:k1])  # 0% ~ 60%   -> train
        val.append(perm[k1:k2])  # 60% ~ 80%  -> validation
        test.append(perm[k2:])   # 20% ~ 100%   -> test
    return x, edge_index, y, torch.cat(train), torch.cat(val), torch.cat(test) 


if __name__ == "__main__":
    dropout = 0.0
    fanout_list_some = [[5],
                   [5, 7],
                   [5, 7, 2]]

    # IMPORTANT: fanout=None  -> full neighborhood (keeps author degree).
    fanout_list_all = [[None],
                       [None, None],
                       [None, None, None]]
    

    af_list = ["gaussian", "zero"] # type of author feature

    simple_mode_list = [True, False] # True: simple SAGE with no dropout and last layer (hidden_dim, 2)
                              # False: more complex SAGE, with relu, dropout and 
                              #                          a small network to predict the label of paper
    sample_some_neighbor_mode = [False] # [True, False]

    for af in af_list: # author feature
        for simple_mode in simple_mode_list:
            for sample_some_neighbor in sample_some_neighbor_mode:
                print(f"\n=== author_feat={af!r}")
                print(f"=== simple_mode={simple_mode!r}")
                print(f"=== simple_some_neighbor={sample_some_neighbor!r}")

                if sample_some_neighbor:
                    fanout_list = fanout_list_some
                else:
                    fanout_list = fanout_list_all
        

                train_acc_L_list = []
                val_acc_L_list = []
                test_acc_L_list= []

                for L in (1, 2, 3):
                    x, ei, y, tr, va, te = build_2hop_hard_instance(author_feat=af)

                    train_acc_list, val_acc_list, test_acc_list = train_sage(x, ei, y, tr, va, te, 
                                                                            num_layers=L, 
                                                                            fanout = fanout_list[L-1], 
                                                                            eval_fanout=None,
                                                                            device="cuda",
                                                                            dropout=dropout,
                                                                            simple_mode=simple_mode)
                    

                    train_acc_L_list.append(train_acc_list)
                    val_acc_L_list.append(val_acc_list)
                    test_acc_L_list.append(test_acc_list)
                
                

                tag = ""
                tag += af
                tag += "-some" if sample_some_neighbor else "-all"
                tag += "-simple" if simple_mode else ""
                save_path = f"../result/2-hop-{tag}.png"


                drawGraph(train_acc_L_list, val_acc_L_list, test_acc_L_list,
                        save_path=save_path)
