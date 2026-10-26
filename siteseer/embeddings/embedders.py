"""Frozen language-model embedders, shared by training, evaluation and inference.

ESM2Embedder: fair-esm ESM-2 at any layer, optionally standardised per dimension (intermediate layers lack the final
LayerNorm). ESMCEmbedder: ESMC-6B, last hidden state. Both return [L, d] float32 on CPU without BOS/EOS.
Presets: esm2_650m_l30 (main model), esm2_650m, esm2_3b, esmc_6b.
"""
from typing import Dict, Optional

import torch

MAX_SEQ_LEN = 1022          # ESM-2's position limit (1024 minus BOS/EOS); used for every encoder
ESMC_MODEL_ID = "biohub/ESMC-6B"
# Snapshot used for the reported runs. The ESMC stage-1 cache was extracted in July 2026 from snapshot 45b0fa5d,
# which reproduces with this one at metric level (same macro AUPRC on a 4,000/1,000-protein linear screen).
ESMC_REVISION = "af1602ba7406f521b11bf8f81d52af378cde09e4"
MAIN_NORM_STATS = "data/embeddings_goliath_esm2_l30/norm_stats.pt"


def load_norm_stats(path: Optional[str]):
    """(mean, std) per dimension, or None for no standardisation."""
    if not path:
        return None
    st = torch.load(path, map_location="cpu", weights_only=False)
    return st["mean"].float(), st["std"].float().clamp_min(1e-6)


def apply_norm(emb: torch.Tensor, stats) -> torch.Tensor:
    return emb if stats is None else (emb - stats[0]) / stats[1]


def load_esm2(device: torch.device, name: str = "esm2_t33_650M_UR50D"):
    import esm
    print(f"[*] loading ESM-2 ({name})...")
    model, alphabet = getattr(esm.pretrained, name)()
    return model.to(device).eval(), alphabet.get_batch_converter()


def load_esmc_tokenizer():
    """ESMC tokenizer via transformers if available, else esm's EsmSequenceTokenizer (same <cls> ... <eos> framing)."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(ESMC_MODEL_ID, revision=ESMC_REVISION, trust_remote_code=True)
    except (ValueError, OSError) as e:
        print(f"[*] AutoTokenizer unavailable for {ESMC_MODEL_ID} ({type(e).__name__}); using EsmSequenceTokenizer")
    from esm.tokenization import EsmSequenceTokenizer
    return EsmSequenceTokenizer()


def load_esmc(device: torch.device):
    from transformers import AutoModel
    print(f"[*] loading {ESMC_MODEL_ID}...")
    model = AutoModel.from_pretrained(ESMC_MODEL_ID, revision=ESMC_REVISION, dtype=torch.float16,
                                      trust_remote_code=True).eval().to(device)
    return model, load_esmc_tokenizer()


class Embedder:
    name = ""
    dim = 0
    max_len = MAX_SEQ_LEN

    def load(self, device: torch.device) -> "Embedder":
        raise NotImplementedError

    def embed(self, sequence: str) -> torch.Tensor:
        raise NotImplementedError

    def _check(self, sequence: str):
        if len(sequence) > self.max_len:
            raise ValueError(f"sequence longer than {self.max_len}; truncate first")


class ESM2Embedder(Embedder):
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", layer: Optional[int] = None,
                 norm_stats: Optional[str] = None, name: str = "esm2"):
        self.model_name, self.norm_stats_path, self.name = model_name, norm_stats, name
        self.layer = layer if layer is not None else (36 if "t36" in model_name else 33)
        self.dim = 2560 if "t36" in model_name else 1280

    def load(self, device):
        self.device = device
        self.model, self.batch_converter = load_esm2(device, self.model_name)
        self.stats = load_norm_stats(self.norm_stats_path)
        return self

    @torch.no_grad()
    def embed(self, sequence):
        self._check(sequence)
        _, _, tokens = self.batch_converter([("prot", sequence)])
        out = self.model(tokens.to(self.device), repr_layers=[self.layer])
        emb = out["representations"][self.layer][0, 1:len(sequence) + 1].float().cpu()
        return apply_norm(emb, self.stats)


class ESMCEmbedder(Embedder):
    name = "esmc_6b"
    dim = 2560

    def load(self, device):
        self.device = device
        self.model, self.tokenizer = load_esmc(device)
        return self

    @torch.no_grad()
    def embed(self, sequence):
        self._check(sequence)
        inputs = {k: v.to(self.device) for k, v in self.tokenizer(sequence, return_tensors="pt").items()}
        return self.model(**inputs).last_hidden_state[0, 1:len(sequence) + 1].float().cpu()


PRESETS: Dict[str, dict] = {
    "esm2_650m_l30": dict(cls=ESM2Embedder, model_name="esm2_t33_650M_UR50D", layer=30, norm_stats=MAIN_NORM_STATS),
    "esm2_650m": dict(cls=ESM2Embedder, model_name="esm2_t33_650M_UR50D", layer=33),
    "esm2_3b": dict(cls=ESM2Embedder, model_name="esm2_t36_3B_UR50D", layer=36),
    "esmc_6b": dict(cls=ESMCEmbedder),
}


def get_embedder(name: str, norm_stats: Optional[str] = None) -> Embedder:
    """A named preset; norm_stats overrides the preset's standardisation file (e.g. a different data root)."""
    if name not in PRESETS:
        raise KeyError(f"unknown embedder {name!r}; available: {sorted(PRESETS)}")
    spec = dict(PRESETS[name])
    cls = spec.pop("cls")
    if cls is ESMCEmbedder:
        return ESMCEmbedder()
    if norm_stats is not None:
        spec["norm_stats"] = norm_stats
    return cls(name=name, **spec)


def embedder_from_config(data_cfg: dict, encoder: str) -> Embedder:
    """The live embedder a training config describes: encoder 'esm_raw' (ESM-2: data.esm_model, data.esm_layer,
    data.norm_stats) or 'esmc_raw' (ESMC-6B)."""
    if encoder == "esmc_raw":
        return ESMCEmbedder()
    if encoder == "esm_raw":
        return ESM2Embedder(model_name=data_cfg.get("esm_model", "esm2_t33_650M_UR50D"),
                            layer=data_cfg.get("esm_layer", 33), norm_stats=data_cfg.get("norm_stats"))
    raise ValueError(f"unknown encoder {encoder!r} (expected 'esm_raw' or 'esmc_raw')")
