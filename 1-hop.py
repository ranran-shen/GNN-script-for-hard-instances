import torch
from SAGE_engine import train_sage   # the graph-agnostic engine
from SAGE_engine import drawGraph


# avoid isolated paper nodes
def build_1hop_hard_instance(num_papers=5000, feat_dim=32,
                             rate0=2.0, rate1=15.0, author_feat="gaussian", seed=0):
    g = torch.Generator().manual_seed(seed)

    # paper labels: first half -> 0, second half -> 1
    half = num_papers // 2
    y_paper = torch.cat([torch.zeros(half), torch.ones(num_papers - half)]).long()

    # number of authors per paper ~ Poisson(rate by label)
    rates = torch.full((num_papers,), rate0)
    rates[y_paper == 1] = rate1
    deg = torch.poisson(rates, generator=g).long()      # authors per paper

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
    # for af in ("gaussian", "zero"):
    # # # for af in ("gaussian"):
    # # for af in ("zero"):
    #     print(f"\n=== author_feat={af!r}")
    af_list = ["gaussian", "zero"]
    mode_list = [True, False]

    for af in af_list:
        for simple_mode in mode_list:
            print(f"\n=== simple_mode={simple_mode!r}")

            # IMPORTANT: fanout=None  -> full neighborhood (keeps author degree).
            fanout_list = [[5],
                            [5, 7],
                            [5, 7, 2]]
            
            train_acc_L_list = []
            val_acc_L_list = []
            test_acc_L_list= []

            for L in (1, 2, 3):
                # N = num of papers + num of authors
                # x: (N, feat_dim)
                # ei: [src, dst], (src[i], dst[i]) is an edge
                # y: (N), y[:num_papers] = y_paper \in {0, 1}
                x, ei, y, tr, va, te = build_1hop_hard_instance(author_feat=af)

                train_acc_list, val_acc_list, test_acc_list = train_sage(x, ei, y, tr, va, te, 
                                                                        num_layers=L, 
                                                                        fanout = fanout_list[L-1], 
                                                                        eval_fanout=None,
                                                                        device="cuda",
                                                                        simple_mode=simple_mode)
                

                train_acc_L_list.append(train_acc_list)
                val_acc_L_list.append(val_acc_list)
                test_acc_L_list.append(test_acc_list)
            
            if simple_mode:
                save_path = "/home/srr/gnn-project/synthetic/June/results/1-hop/1-hop-"+af+"-s"+".png"
            else:
                save_path = "/home/srr/gnn-project/synthetic/June/results/1-hop/1-hop-"+af+".png"
            
            drawGraph(train_acc_L_list, val_acc_L_list, test_acc_L_list,
                    save_path=save_path)
