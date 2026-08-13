import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None


class TTTSWACollector:
    """Collects per-layer TTT/SWA gate activations via forward hooks and
    reports summary stats to wandb.

    Renamed from GateSWACollector to match what demo_viser.py imports.
    Also made robust to TTT/SWA being disabled (model.ttt_gate_projs /
    model.swa_gate_projs can be None), and given the log_to_wandb() method
    that demo_viser.py actually calls.
    """

    def __init__(self, model):
        self.model = model
        self.window_idx = 0
        self.records = []
        self.handles = []

        ttt_gate_projs = getattr(model, "ttt_gate_projs", None)
        swa_gate_projs = getattr(model, "swa_gate_projs", None)

        if ttt_gate_projs is not None:
            for i, proj in enumerate(ttt_gate_projs):
                self.handles.append(proj.register_forward_hook(self._make_hook("ttt", i)))
        if swa_gate_projs is not None:
            for i, proj in enumerate(swa_gate_projs):
                self.handles.append(proj.register_forward_hook(self._make_hook("swa", i)))

    def _make_hook(self, kind, layer_idx):
        def hook(module, inp, out):
            gate = F.silu(out).abs().mean().item()
            self.records.append({
                "window": self.window_idx,
                "kind": kind,
                "layer": layer_idx,
                "gate_mean": gate,
            })
        return hook

    def advance_window(self):
        self.window_idx += 1

    def log_to_wandb(self):
        """Reduce collected per-layer gate records into wandb scalars.

        Note: demo_viser.py currently never calls advance_window(), so
        self.window_idx stays 0 for every record collected during a single
        model(...) call -- these numbers are aggregated across all windows
        of that call, not broken out per window. For a true per-window
        breakdown, use raw_model_predictions["per_window_gate_records"]
        (populated directly by Pi3.forward), which is window-accurate.
        """
        if wandb is None or wandb.run is None or not self.records:
            return

        ttt_records = [r for r in self.records if r["kind"] == "ttt"]
        swa_records = [r for r in self.records if r["kind"] == "swa"]

        log_dict = {}

        if ttt_records:
            ttt_vals = [r["gate_mean"] for r in ttt_records]
            log_dict["gate/ttt_mean"] = sum(ttt_vals) / len(ttt_vals)
            per_layer = {}
            for r in ttt_records:
                per_layer.setdefault(r["layer"], []).append(r["gate_mean"])
            for layer_idx, vals in sorted(per_layer.items()):
                log_dict[f"gate/ttt_layer_{layer_idx}"] = sum(vals) / len(vals)

        if swa_records:
            swa_vals = [r["gate_mean"] for r in swa_records]
            log_dict["gate/swa_mean"] = sum(swa_vals) / len(swa_vals)
            per_layer = {}
            for r in swa_records:
                per_layer.setdefault(r["layer"], []).append(r["gate_mean"])
            for layer_idx, vals in sorted(per_layer.items()):
                log_dict[f"gate/swa_layer_{layer_idx}"] = sum(vals) / len(vals)

        if log_dict:
            wandb.log(log_dict)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# Backwards-compatible alias in case other code still imports the old name.
GateSWACollector = TTTSWACollector