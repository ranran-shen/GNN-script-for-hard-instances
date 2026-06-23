import torch
from SAGE_engine import train_sage   # the graph-agnostic engine
from SAGE_engine import drawGraph


def expand_groups_to_papers(group_ids, paper_start, group_size):
    """Given group ids, return all paper node ids in these groups."""
    out = []
    for grp in group_ids.tolist():
        base = paper_start + grp * group_size
        out.append(torch.arange(base, base + group_size, dtype=torch.long))
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long)


def split_groups(num_groups, paper_start, group_size, generator,
                 train_ratio=0.6, val_ratio=0.2):
    """Split groups, then expand each split into paper node ids."""
    group_perm = torch.randperm(num_groups, generator=generator)

    n_train = int(train_ratio * num_groups)
    n_val = int(val_ratio * num_groups)

    train_groups = group_perm[:n_train]
    val_groups = group_perm[n_train:n_train + n_val]
    test_groups = group_perm[n_train + n_val:]

    train_ids = expand_groups_to_papers(train_groups, paper_start, group_size)
    val_ids = expand_groups_to_papers(val_groups, paper_start, group_size)
    test_ids = expand_groups_to_papers(test_groups, paper_start, group_size)

    return train_ids, val_ids, test_ids


def make_8regular_10():
    """A 10-vertex 8-regular template = K_10 minus a perfect matching.

    Each vertex has degree 8; |E| = 10*8/2 = 40.
    Removing the matching {(i, i+5) : i in 0..4} from K_10 leaves a SIMPLE
    8-regular graph. Each of the 40 edges later becomes ONE author linking the
    two papers (= the two endpoint vertices).

    The template is SIMPLE, so inside a label-0 group any two papers share AT
    MOST ONE author  =>  NO bipartite 4-cycle (paper-author-paper-author).
    """
    matching = {(i, i + 5) for i in range(5)}
    edges = []
    for i in range(10):
        for j in range(i + 1, 10):
            if (i, j) in matching:
                continue
            edges.append((i, j))
    assert len(edges) == 40
    return edges


def build_4cycle_hard_instance(num_papers_per_class=2400,
                               group1_size=2, authors_per_group1=8,
                               group0_size=10,
                               feat_dim=32, author_feat="gaussian",
                               paper_feat="gaussian",
                               seed=0):
    """4-cycle hard instance.

    label 1 (HAS 4-cycles): papers grouped in pairs; the 2 papers of a group
        share `authors_per_group1` = 8 authors  ->  a K_{2,8} block.
        Any two of the 8 shared authors close a 4-cycle  p1 - a - p2 - a' - p1.
        groups = 2400/2 = 1200 ; authors = 1200*8 = 9600.

    label 0 (NO 4-cycles): papers grouped in 10; inside a group we lay down a
        10-vertex 8-regular template and turn each of its 40 edges into ONE
        author linking the two endpoint papers (graph subdivision / incidence).
        groups = 2400/10 = 240 ; authors = 240*40 = 9600.

    Invariants that make this a CLEAN 4-cycle test (nothing else leaks the label):
        * every paper  has degree 8   (label1: 8 shared authors ; label0: 8-regular)
        * every author has degree 2   (label1: the 2 papers of its pair ;
                                       label0: the 2 endpoints of its template edge)
        * features are frozen random Gaussian (or zero for authors)
      => degree- and feature-statistics are IDENTICAL across the two classes; the
         ONLY discriminating signal is 4-cycle participation, i.e. whether two of
         a paper's authors lead back to a common second paper (co-neighbour
         multiplicity >= 2).

    Node id layout:  label0 papers | label1 papers | label1 authors | label0 authors
    Returns: x, edge_index, y, train_ids, val_ids, test_ids   (same API as 2-hop).
    """
    g = torch.Generator().manual_seed(seed)

    num_papers = 2 * num_papers_per_class            # 4800
    LBL0_P0 = 0                                       # label-0 papers start here
    LBL1_P0 = num_papers_per_class                    # label-1 papers start here

    # ---- label 1 sizing ----
    assert num_papers_per_class % group1_size == 0
    num_groups1 = num_papers_per_class // group1_size        # 1200
    num_authors1 = num_groups1 * authors_per_group1          # 9600

    # ---- label 0 sizing ----
    template = make_8regular_10()
    edges_per_group0 = len(template)                         # 40
    assert num_papers_per_class % group0_size == 0
    num_groups0 = num_papers_per_class // group0_size        # 240
    num_authors0 = num_groups0 * edges_per_group0            # 9600

    num_authors = num_authors1 + num_authors0                # 19200
    N = num_papers + num_authors                             # 24000

    AUTH1_0 = num_papers                                     # label-1 authors start
    AUTH0_0 = num_papers + num_authors1                      # label-0 authors start

    # edges (both directions, kept symmetric like 2-hop)
    src, dst = [], []

    # ---- label 1: K_{2,8} per group  (creates 4-cycles) ----
    for grp in range(num_groups1):
        p_base = LBL1_P0 + grp * group1_size
        papers = range(p_base, p_base + group1_size)         # the 2 papers
        a_base = AUTH1_0 + grp * authors_per_group1
        for a in range(a_base, a_base + authors_per_group1): # the 8 shared authors
            for p in papers:
                src += [p, a]; dst += [a, p]

    # ---- label 0: subdivision of the 8-regular template (no 4-cycles) ----
    for grp in range(num_groups0):
        p_base = LBL0_P0 + grp * group0_size
        a_base = AUTH0_0 + grp * edges_per_group0
        for e_idx, (u, v) in enumerate(template):
            a = a_base + e_idx
            pu, pv = p_base + u, p_base + v
            src += [pu, a, pv, a]; dst += [a, pu, a, pv]

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # features: frozen random Gaussian for everyone
    x = torch.randn(N, feat_dim, generator=g)
    if author_feat == "zero":
        x[num_papers:] = 0.0
        print("author's feature: 0!!!!!")
    elif author_feat == "one":
        x[num_papers:] = 1.0
        print("author's feature: 1.0!!!!!")

    if paper_feat == "zero":
        x[:num_papers] = 0.0
        print("paper's feature: 0!!!!!")
    elif paper_feat == "one":
        x[:num_papers] = 1.0
        print("paper's feature: 1.0!!!!!")

    # labels (only papers carry a real label; authors get 0 as a placeholder)
    y = torch.zeros(N, dtype=torch.long)
    y[LBL1_P0:num_papers] = 1
    y_paper = y[:num_papers]

    # # split paper nodes into train/val/test, balanced per class (60/20/20)
    # train, val, test = [], [], []
    # for c in (0, 1):
    #     ids = (y_paper == c).nonzero(as_tuple=True)[0]
    #     n = ids.numel()
    #     perm = ids[torch.randperm(n, generator=g)]
    #     k1, k2 = int(0.6 * n), int(0.8 * n)
    #     train.append(perm[:k1])   # 0%   ~ 60%  -> train
    #     val.append(perm[k1:k2])   # 60%  ~ 80%  -> validation
    #     test.append(perm[k2:])    # 80%  ~ 100% -> test
    # return x, edge_index, y, torch.cat(train), torch.cat(val), torch.cat(test)
    # ---- group-level split: no constructed block crosses train/val/test ----

    # label 0 groups: each has group0_size = 10 papers
    tr0, va0, te0 = split_groups(
        num_groups=num_groups0,
        paper_start=LBL0_P0,
        group_size=group0_size,
        generator=g,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    # label 1 groups: each has group1_size = 2 papers
    tr1, va1, te1 = split_groups(
        num_groups=num_groups1,
        paper_start=LBL1_P0,
        group_size=group1_size,
        generator=g,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    train = torch.cat([tr0, tr1])
    val = torch.cat([va0, va1])
    test = torch.cat([te0, te1])

    # optional: shuffle within each split
    train = train[torch.randperm(train.numel(), generator=g)]
    val = val[torch.randperm(val.numel(), generator=g)]
    test = test[torch.randperm(test.numel(), generator=g)]

    return x, edge_index, y, train, val, test


if __name__ == "__main__":
    dropout = 0.5

    # paper degree = 8 (both classes), author degree = 2 (both classes).
    # NOTE: author-hops only have 2 neighbours, so a fanout of 2 there is ~full
    #       (khop_sample draws WITH replacement, so it can still miss one of them).
    fanout_list_some = [[5],
                        [5, 1],
                        [5, 1, 1]]

    # fanout=None -> full neighbourhood (degree-preserving). This is the clean run.
    fanout_list_all = [[None],
                       [None, None],
                       [None, None, None],
                       [None, None, None, None]]
    
    # layer_list = [1,2,3]
    layer_list = [2,3,4]
    # start = 2

    # af_list = ["gaussian", "zero"] # type of author feature
    # af_list = ["gaussian"] # type of author feature
    af_list = ["gaussian", "one"]
    # pp_list = ["gaussian", "zero"] # paper feature
    pp_list = ["gaussian", "one"] # paper feature

    # simple_mode_list = [True, False]          # True: simple SAGE (ReLU only, last layer -> 2)
                                              # False: + dropout + small readout head
    simple_mode_list = [False]
    # sample_some_neighbor_mode = [False, True] # False -> full neighbourhood, True -> subsample
    sample_some_neighbor_mode = [False] # False -> full neighbourhood, True -> subsample

      
    for ppf in pp_list: 
        for af in af_list:
            for simple_mode in simple_mode_list:
                for sample_some_neighbor in sample_some_neighbor_mode:
                    print(f"\n=== paper_feat={ppf!r}")
                    print(f"=== author_feat={af!r}")
                    print(f"=== simple_mode={simple_mode!r}")
                    print(f"=== sample_some_neighbor={sample_some_neighbor!r}")

                    if sample_some_neighbor:
                        fanout_list = fanout_list_some
                    else:
                        fanout_list = fanout_list_all

                    train_acc_L_list = []
                    val_acc_L_list = []
                    test_acc_L_list = []

                    for L in layer_list:
                        # N = num of papers + num of authors
                        # x: (N, feat_dim)
                        # ei: [src, dst], (src[i], dst[i]) is an edge
                        # y: (N), y[:num_papers] = y_paper \in {0, 1}
                        x, ei, y, tr, va, te = build_4cycle_hard_instance(author_feat=af,
                                                                          paper_feat=ppf)

                        train_acc_list, val_acc_list, test_acc_list = train_sage(
                            x, ei, y, tr, va, te,
                            num_layers=L,
                            fanout=fanout_list[L - 1],
                            eval_fanout=None,
                            device="cuda",
                            dropout=dropout,
                            simple_mode=simple_mode)

                        train_acc_L_list.append(train_acc_list)
                        val_acc_L_list.append(val_acc_list)
                        test_acc_L_list.append(test_acc_list)

                    tag = ""
                    tag = tag + 'pf=' + ppf
                    tag = tag + '-af=' + af
                    tag += "-some" if sample_some_neighbor else "-all"
                    tag += "-simple" if simple_mode else ""
                    save_path = f"./results/4-cycle/split-group/layer4/drop-{tag}.png"

                    drawGraph(train_acc_L_list, val_acc_L_list, test_acc_L_list,layer_list=layer_list,
                            save_path=save_path)
