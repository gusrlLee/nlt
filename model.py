"""Small Torch-only hash-grid baselines with identical input features."""
import math

import torch
from torch import nn


class HashGrid(nn.Module):
    def __init__(self, bounds):
        super().__init__()
        assert bounds.shape == (2, 3) and torch.isfinite(bounds).all()
        assert (bounds[1] > bounds[0]).all()
        self.register_buffer("bounds", bounds.detach().clone().float())
        self.register_buffer("resolutions", torch.tensor([
            round(16 * math.exp(math.log(128 / 16) * i / 7)) for i in range(8)]))
        self.register_buffer("corners", torch.tensor([
            [x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]))
        self.tables = nn.ModuleList([nn.Embedding(2**14, 2) for _ in range(8)])
        for table in self.tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)

    def forward(self, position):
        p = ((position - self.bounds[0]) / (self.bounds[1] - self.bounds[0])).clamp(0, 1)
        levels = []
        for resolution, table in zip(self.resolutions, self.tables):
            scaled = p * resolution
            base = scaled.floor().long()
            fraction = scaled - base
            vertices = base[:, None, :] + self.corners
            index = (vertices[..., 0] ^ (vertices[..., 1] * 2654435761)
                     ^ (vertices[..., 2] * 805459861)) % table.num_embeddings
            weights = torch.where(self.corners.bool(), fraction[:, None, :],
                                  1 - fraction[:, None, :]).prod(dim=-1)
            levels.append((table(index) * weights[..., None]).sum(dim=1))
        return torch.cat(levels, dim=-1)


class RadianceModel(nn.Module):
    def __init__(self, kind, bounds):
        super().__init__()
        assert kind in ("mlp", "gru")
        self.kind = kind
        self.grid = HashGrid(bounds)
        if kind == "mlp":
            self.core = nn.Sequential(nn.Linear(26, 64), nn.ReLU(),
                                      nn.Linear(64, 64), nn.ReLU())
        else:
            self.core = nn.GRUCell(26, 64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                  nn.Linear(32, 3), nn.Softplus())

    def _recurrent_step(self, encoded, mask, hidden):
        hidden = torch.where(mask[:, None], self.core(encoded, hidden), hidden)
        return self.head(hidden) * mask[:, None], hidden

    def step(self, features, mask, hidden=None):
        """One surface bounce; callers retain one hidden row per original ray."""
        assert self.kind == "gru"
        assert features.ndim == 2 and features.shape[1] == 13
        assert mask.shape == features.shape[:1] and mask.dtype == torch.bool
        assert torch.isfinite(features).all()
        if hidden is None:
            hidden = features.new_zeros((len(features), 64))
        assert hidden.shape == (len(features), 64) and torch.isfinite(hidden).all()
        assert hidden.device == features.device and hidden.dtype == features.dtype
        encoded = features.new_zeros((len(features), 26))
        valid = features[mask]
        encoded[mask] = torch.cat((self.grid(valid[:, :3]), valid[:, 3:]), dim=-1)
        return self._recurrent_step(encoded, mask, hidden)

    def forward(self, features, mask):
        assert features.ndim == 3 and features.shape[-1] == 13
        assert mask.shape == features.shape[:2] and mask.dtype == torch.bool
        assert features.shape[1] > 0 and torch.isfinite(features).all()
        assert not (mask[:, 1:] & ~mask[:, :-1]).any()
        # Each row is one complete ray, with an independent initial state.
        encoded = features.new_zeros((*mask.shape, 26))
        valid = features[mask]
        encoded[mask] = torch.cat((self.grid(valid[:, :3]), valid[:, 3:]), dim=-1)
        if self.kind == "mlp":
            prediction = self.head(self.core(encoded))
        else:
            h = features.new_zeros((len(features), 64))
            predictions = []
            for d in range(features.shape[1]):
                prediction, h = self._recurrent_step(encoded[:, d], mask[:, d], h)
                predictions.append(prediction)
            prediction = torch.stack(predictions, dim=1)
        return prediction * mask[..., None]


def masked_mse(prediction, target, mask):
    assert prediction.shape == target.shape == (*mask.shape, 3)
    assert torch.isfinite(prediction).all() and torch.isfinite(target).all()
    assert mask.any(), "Batch has no valid surface vertices"
    return (prediction[mask] - target[mask]).square().mean()


def self_check(device=None):
    from runtime import select_device
    device = select_device() if device is None else torch.device(device)
    torch.manual_seed(42)
    bounds = torch.tensor([[-1., 0., -1.], [1., 2., 1.]])
    x = torch.rand(3, 4, 13, device=device)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False],
                         [False, False, False, False]], device=device)
    for kind in ("mlp", "gru"):
        model = RadianceModel(kind, bounds).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        y = model(x, mask)
        assert y.shape == (3, 4, 3) and torch.isfinite(y).all()
        assert (y[~mask] == 0).all()
        assert torch.allclose(y[0], model(x[:1], mask[:1])[0], atol=1e-6)
        assert torch.allclose(y[:, :2], model(x[:, :2], mask[:, :2]), atol=1e-6)
        changed = x.clone()
        changed[1] += 1
        assert torch.allclose(y[0], model(changed, mask)[0], atol=1e-6)
        changed = x.clone()
        changed[0, 0] += .5
        if kind == "gru":
            assert not torch.allclose(y[0, 1], model(changed, mask)[0, 1])
            hidden, steps = None, []
            for d in range(x.shape[1]):
                previous = hidden
                prediction, hidden = model.step(x[:, d], mask[:, d], hidden)
                steps.append(prediction)
                if previous is not None:
                    assert torch.equal(hidden[~mask[:, d]], previous[~mask[:, d]])
            assert torch.allclose(y, torch.stack(steps, dim=1), atol=1e-6)
            empty, unchanged = model.step(
                x[:, 0], torch.zeros(3, dtype=torch.bool, device=device), hidden)
            assert (empty == 0).all() and torch.equal(unchanged, hidden)
        loss = masked_mse(y, torch.zeros_like(y), mask)
        assert torch.allclose(loss, y[mask].square().mean())
        loss.backward()
        assert model.grid.tables[0].weight.grad is not None
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
    print(f"PASS ({device}): shapes, masks, gradients, optimizer, independent rays, "
          "recurrent history and streaming steps")


if __name__ == "__main__":
    self_check()
