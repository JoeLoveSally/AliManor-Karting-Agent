import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from karting_agent.model.base import build_model


@pytest.mark.parametrize("architecture", ["mobilenet_v3_small", "resnet18"])
def test_temporal_model_accepts_nine_channels(architecture: str) -> None:
    model = build_model(architecture, frame_stack=3, pretrained=False)
    inputs = torch.zeros(2, 9, 224, 224)

    logits = model(inputs)

    assert logits.shape == (2,)
