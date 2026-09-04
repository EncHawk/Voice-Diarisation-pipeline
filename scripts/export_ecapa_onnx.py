"""Export SpeechBrain ECAPA-TDNN to ONNX for runtime without torch.

Requires: torch, speechbrain, torchaudio (uv sync --group export)
"""
from __future__ import annotations

import pathlib

MODELS_DIR = pathlib.Path("models")
OUT = MODELS_DIR / "ecapa.onnx"


def main():
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as e:
        print(f"[export] missing dependency: {e}")
        print("  Run: uv sync --group export  (or: uv pip install torch speechbrain torchaudio)")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"[export] {OUT} already exists, skipping (delete to re-export)")
        return

    print("[export] loading SpeechBrain spkrec-ecapa-voxceleb ...")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODELS_DIR / "sb_cache"))
    classifier.eval()

    # Wrap for ONNX export: takes (1, n_samples) float32
    sample_len = 48000  # 3s @16k

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, wav):
            # wav: (1, n)
            # speechbrain expects (1, n) and returns (1, 1, 192)
            emb = self.model.encode_batch(wav)
            return emb.squeeze(1)

    wrapper = Wrapper(classifier)
    wrapper.eval()
    dummy = torch.randn(1, sample_len)

    print(f"[export] exporting to {OUT} ...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(OUT),
        input_names=["wav"],
        output_names=["embedding"],
        dynamic_axes={"wav": {1: "length"}},
        opset_version=17,
    )
    print(f"[export] done -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
