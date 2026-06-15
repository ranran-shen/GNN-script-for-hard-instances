import torch
import torch.nn.functional as F

# reuse the from-scratch primitives unchanged
from SAGE_engine import build_adj, khop_sample, SAGE, drawGraph


# ======================================================================
# 1) STRUCTURE  --  built ONCE, shared by every hard instance
#    (topology + labels + paper splits; NO features here)
# ======================================================================
def build_2hop_structure(num_papers=5000, num_prolific=100, num_nonprolific=1900,
                         deg=8, seed=0):
    g = torch.Generator().manual_seed(seed)

    num_authors = num_prolific + num_nonprolific
    N = num_papers + num_authors

    PROL0 = num_papers                  # prolific author ids:    [PROL0, NONP0)
    NONP0 = num_papers + num_prolific    # non-prolific author ids: [NONP0, N)

    half = num_papers // 2
    y_paper = torch.cat([torch.zeros(half),
                         torch.ones(num_papers - half)]).long()

    prolific    = torch.arange(PROL0, PROL0 + num_prolific)
    nonprolific = torch.arange(NONP0, NONP0 + num_nonprolific)

    # each paper -> `deg` DISTINCT authors from its class pool (both directions)
    src, dst = [], []
    for p in range(num_papers):
        pool = prolific if y_paper[p].item() == 1 else nonprolific
        sel = pool[torch.randperm(pool.numel(), generator=g)[:deg]]
        for a in sel.tolist():
            src += [p, a]; dst += [a, p]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    y = torch.zeros(N, dtype=torch.long)
    y[:num_papers] = y_paper

    # stratified split + balance assertion (avoids the old seed-collision imbalance)
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

    train, val, test = torch.cat(train), torch.cat(val), torch.cat(test)

    for name, s in [("train", train), ("val", val), ("test", test)]:
        frac1 = (y[s] == 1).float().mean().item()
        assert abs(frac1 - 0.5) < 0.05, f"{name} label imbalance: {frac1:.3f}"

    meta = dict(N=N, num_papers=num_papers, num_prolific=num_prolific,
                num_nonprolific=num_nonprolific, PROL0=PROL0, NONP0=NONP0, deg=deg)
    return edge_index, y, train, val, test, meta # no fearures


# ======================================================================
# 2) FEATURES  --  resampled per instance
#    Authors stay random Gaussian. Because every
#    instance draws fresh features, the prolific / non-prolific *centroids*
#    move each instance, so a shared linear readout cannot memorise them
#    -> the 1-hop centroid leak is closed. Degree (the 2-hop signal) is
#    purely structural and is identical across instances -> still learnable.
# ======================================================================
def sample_features(N, num_papers, feat_dim, g, author_feat="gaussian"):
    x = torch.randn(N, feat_dim, generator=g)
    if author_feat == "zero":            # control only; not needed for inductive
        x[num_papers:] = 0.0
    return x


# ======================================================================
# 3) DIAGNOSTIC  --  confirm the structural signal exists and is the
#    ONLY thing that differs between the two classes.
# ======================================================================
def diagnose_degrees(edge_index, meta):
    N, P, NP0 = meta["N"], meta["PROL0"], meta["NONP0"]
    deg = torch.zeros(N, dtype=torch.long)
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    paper_deg = deg[:P].float()
    prol_deg  = deg[P:NP0].float()
    nonp_deg  = deg[NP0:].float()
    print(f"[diag] paper degree:        {paper_deg.mean():.2f} "
          f"(min {paper_deg.min():.0f}, max {paper_deg.max():.0f})  <- should be flat = deg")
    print(f"[diag] prolific deg:        {prol_deg.mean():.2f} "
          f"(min {prol_deg.min():.0f}, max {prol_deg.max():.0f})")
    print(f"[diag] non-prolific deg:    {nonp_deg.mean():.2f} "
          f"(min {nonp_deg.min():.0f}, max {nonp_deg.max():.0f})")
    print(f"[diag] degree gap (the 2-hop signal): {prol_deg.mean()/max(nonp_deg.mean(),1e-9):.1f}x")


# ======================================================================
# 4) INDUCTIVE TRAINER
#    Same engine as before (khop_sample + SAGE), but features are
#    redrawn every `resample_every` steps, and eval uses a fresh,
#    held-out feature draw the model never trained on.
# ======================================================================
def train_sage_inductive(edge_index, y, train_ids, val_ids, test_ids, meta,
                         num_layers, fanout=None, eval_fanout=None,
                         feat_dim=32, author_feat="gaussian",
                         n_classes=2, hid=32, steps=200, bs=128,
                         lr=1e-2, weight_decay=0, dropout=0.0,
                         device="cpu", eval_every=20, simple_mode=True,
                         resample_every=1):
    torch.manual_seed(0)
    N, num_papers = meta["N"], meta["num_papers"]

    y = y.to(device)

    train_ids = train_ids.to(device)
    val_ids = val_ids.to(device)
    test_ids = test_ids.to(device)

    adj = build_adj(edge_index, N)       # built ONCE -- topology never changes

    # fanout bookkeeping (unchanged from your engine)
    train_fanouts = [fanout] * num_layers if (isinstance(fanout, int) or fanout is None) else fanout
    eval_fanouts  = [eval_fanout] * num_layers if (isinstance(eval_fanout, int) or eval_fanout is None) else eval_fanout
    assert len(train_fanouts) == num_layers

    model = SAGE(feat_dim, hid, n_classes, num_layers, dropout, simple_mode=simple_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    g_feat = torch.Generator().manual_seed(1)   # TRAIN feature stream (advances every resample)
    g_samp = torch.Generator().manual_seed(2)   # neighbour-sampling / batch stream

    def eval_split(ids, feat_seed, samp_seed):
        """Evaluate on a FRESH feature draw (unseen during training)."""
        model.eval()
        gf = torch.Generator().manual_seed(feat_seed)
        gs = torch.Generator().manual_seed(samp_seed)
        x = sample_features(N, num_papers, feat_dim, gf, author_feat).to(device)
        with torch.no_grad():
            sx, ei, sl, n = khop_sample(ids.cpu(), adj, x, eval_fanouts, gs)
            pred = model(sx.to(device), ei.to(device), sl.to(device), n).argmax(1)
            return (pred == y[ids]).float().mean().item()

    train_acc_list, val_acc_list, test_acc_list = [], [], []
    x = sample_features(N, num_papers, feat_dim, g_feat, author_feat).to(device)

    for step in range(steps):
        if step % resample_every == 0:           # NEW features -> nothing stable to memorise
            x = sample_features(N, num_papers, feat_dim, g_feat, author_feat).to(device)

        model.train()
        opt.zero_grad()

        batch = train_ids[torch.randperm(train_ids.numel(), generator=g_samp)[:bs]]
        sx, ei, sl, n = khop_sample(batch.cpu(), adj, x, train_fanouts, g_samp)
        logits = model(sx.to(device), ei.to(device), sl.to(device), n)
        loss = F.cross_entropy(logits, y[batch])
        loss.backward(); opt.step()

        with torch.no_grad():
            train_acc_list.append((logits.argmax(1) == y[batch]).float().mean().item())
        val_acc_list.append(eval_split(val_ids,  feat_seed=7, samp_seed=17))
        test_acc_list.append(eval_split(test_ids, feat_seed=8, samp_seed=18))

        if step % eval_every == 0:
            print(f"  step {step:3d} | loss {loss.item():.3f} | "
                  f"train {train_acc_list[-1]:.3f} | val {val_acc_list[-1]:.3f} | test {test_acc_list[-1]:.3f}")

    return train_acc_list, val_acc_list, test_acc_list


# ======================================================================
# 5) SWEEP
# ======================================================================
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FEAT_DIM = 32
    dropout=0.0

    # structure is built ONCE and shared by every config / every instance
    edge_index, y, tr, va, te, meta = build_2hop_structure()
    diagnose_degrees(edge_index, meta)

    # fanout=None  -> full neighbourhood, PRESERVES author degree (the correct setting)
    # fixed fanout -> destroys the degree signal (control: even 2-layer should fail)
    fanout_all  = [[None], [None, None], [None, None, None]]
    fanout_some = [[5],    [5, 7],       [5, 7, 2]]


    for af in ["gaussian"]:
        for simple_mode in [True, False]:
            # for sample_some, fanout_list in [(False, fanout_all), (True, fanout_some)]:
            for sample_some, fanout_list in [(False, fanout_all)]:
                print(f"\n=== author_feat={af} | simple={simple_mode} | sample_some={sample_some}")
                trA, vaA, teA = [], [], []
                for L in (1, 2, 3):
                    print(f"--- L={L}")
                    a, b, c = train_sage_inductive(
                        edge_index, y, tr, va, te, meta,
                        num_layers=L, fanout=fanout_list[L - 1], eval_fanout=None,
                        feat_dim=FEAT_DIM, author_feat=af,dropout=dropout,
                        device=DEVICE, simple_mode=simple_mode)
                    trA.append(a); vaA.append(b); teA.append(c)

                tag = af
                tag += "-some" if sample_some else "-all"
                tag += "-simple" if simple_mode else ""
                save = f"../result/2-hop-inductive-{tag}.png"
                drawGraph(trA, vaA, teA, save_path=save)
                print(f"saved -> {save}")
