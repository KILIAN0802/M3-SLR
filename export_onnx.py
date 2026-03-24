import torch
from modelling.Uniformer import UsimKD
from utils.misc import load_config
import argparse

class ExportModel(torch.nn.Module):
    """
    Wrapper để export UsimKD sang ONNX vì UsimKD forward nhận dict
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x shape: [B,3,T,H,W]
        out = self.model.forward({"rgb_center": x})
        return out["logits"]


def main():
    parser = argparse.ArgumentParser("Export UsimKD to ONNX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UsimKD(
        num_classes=cfg["model"]["num_classes"],
        embed_size=cfg["model"]["embed_size"],
        pretraiend=False,
        pretrained_name=None,
        hierarchical_simkd=cfg["model"].get("hierarchical_simkd", False),
        simkd3v=cfg["model"].get("simkd3v", False),
        device=device
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    # Bọc lại
    export_model = ExportModel(model).to(device).eval()

    dummy_input = torch.randn(1, 3, 16, 224, 224).to(device)

    print("🚀 Exporting to:", args.output)
    torch.onnx.export(
        export_model,
        dummy_input,
        args.output,
        opset_version=14,
        do_constant_folding=True,
        input_names=["rgb_center"],
        output_names=["logits"],
        dynamic_axes={"rgb_center": {0: "batch"}, "logits": {0: "batch"}}
    )

    print("✅ ONNX Export thành công:", args.output)


if __name__ == "__main__":
    main()
