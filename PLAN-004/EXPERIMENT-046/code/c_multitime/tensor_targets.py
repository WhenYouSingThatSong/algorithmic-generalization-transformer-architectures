"""Precompute graph topology once; build synchronous labels entirely on device."""
import torch


class Topology:
    def __init__(self, graphs):
        # Training is fixed N32. Equal node counts avoid padded-index aliasing.
        n = len(graphs[0]['nodes'])
        if any(len(g['nodes']) != n for g in graphs):
            raise ValueError('group equal node counts before constructing topology')
        k = max(1, max(len(ps) for g in graphs for ps in g['parents'].values()))
        pos, depths, parents, masks, constants = [], [], [], [], []
        for g in graphs:
            pos.append(g['nodes'])
            depths.append([g['depths'][p] for p in g['nodes']])
            parents.append([g['parents'][p]+[0]*(k-len(g['parents'][p])) for p in g['nodes']])
            masks.append([[True]*len(g['parents'][p])+[False]*(k-len(g['parents'][p])) for p in g['nodes']])
            constants.append([g['constants'].get(p, 0) for p in g['nodes']])
        self.data = dict(pos=torch.tensor(pos), depths=torch.tensor(depths),
                         parents=torch.tensor(parents), parent_mask=torch.tensor(masks, dtype=torch.bool),
                         constants=torch.tensor(constants))

    def batch(self, ids, length, device):
        data = {k: v[ids].to(device) for k, v in self.data.items()}
        return BatchTopology(data, length)


class BatchTopology:
    def __init__(self, data, length):
        self.__dict__.update(data)
        b = self.pos.shape[0]
        self.bi = torch.arange(b, device=self.pos.device)[:, None]
        self.nodes = torch.zeros(b, length, dtype=torch.bool, device=self.pos.device)
        self.nodes.scatter_(1, self.pos, True)

    @torch.no_grad()
    def local(self, current, original):
        """Same rule as c_update.protocol.local_step, including unknown-parent hold."""
        old = current[..., 3].gather(1, self.pos)
        pv = current[..., 3].gather(1, self.parents.flatten(1)).reshape(self.parents.shape)
        ready = (((pv >= 0) & (pv < 23)) | ~self.parent_mask).all(-1)
        sums = (pv * self.parent_mask).sum(-1).remainder(23)
        leaf = ~self.parent_mask.any(-1)
        const = original[..., 3].gather(1, self.constants)
        expected = torch.where(leaf, const, torch.where(ready, sums, old))
        target = original[..., :4].clone()
        target[self.bi, self.pos, 3] = expected
        reference = torch.zeros_like(self.nodes)
        reference.scatter_(1, self.pos, expected != old)
        return target, reference

    @torch.no_grad()
    def teacher(self, initial, original, t):
        """Vectorized old correction labels, provided for parity checks/reuse."""
        ready = torch.zeros_like(self.nodes)
        ready.scatter_(1, self.pos, self.depths <= t)
        return torch.where(ready[..., None], original, initial)
