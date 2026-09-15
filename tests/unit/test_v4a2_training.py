import pytest


def test_v4a2_keeps_ordered_features_until_temporal_mlp() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from karting_agent.model.temporal_mlp_state_conditioned import (
        build_temporal_mlp_state_conditioned_model,
    )

    torch.manual_seed(11)
    model = build_temporal_mlp_state_conditioned_model(
        "mobilenet_v3_small",
        pretrained=False,
        frame_stack=5,
        horizon_count=3,
        visual_feature_dim=16,
        temporal_hidden_dim=12,
        state_embedding_dim=4,
        policy_hidden_dim=8,
    ).eval()
    inputs = torch.randn(2, 5, 3, 64, 64)
    captured_linear_inputs = []

    first_linear = model.temporal_fusion[1]

    def capture_linear_input(_module, args) -> None:
        captured_linear_inputs.append(args[0].detach().clone())

    hook = first_linear.register_forward_pre_hook(capture_linear_input)
    try:
        with torch.inference_mode():
            features = model.encode_sequence(inputs)
            reversed_features = model.encode_sequence(inputs.flip(1))
            switch_logits, future_logits = model(inputs, torch.tensor([0, 1]))
            reversed_switch, reversed_future = model(
                inputs.flip(1), torch.tensor([0, 1])
            )
    finally:
        hook.remove()

    assert features.shape == (2, 5, 16)
    assert torch.allclose(reversed_features, features.flip(1), atol=1e-6, rtol=1e-5)
    assert switch_logits.shape == (2, 3)
    assert future_logits.shape == (2, 3)
    assert reversed_switch.shape == (2, 3)
    assert reversed_future.shape == (2, 3)

    assert len(captured_linear_inputs) == 2
    expected = features.reshape(2, -1)
    expected_reversed = reversed_features.reshape(2, -1)
    assert torch.allclose(captured_linear_inputs[0], expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        captured_linear_inputs[1], expected_reversed, atol=1e-6, rtol=1e-5
    )

    # Reverse time must reverse feature chunks, not average or pool them first.
    chunked = captured_linear_inputs[0].reshape(2, 5, 16)
    reversed_chunked = captured_linear_inputs[1].reshape(2, 5, 16)
    assert torch.allclose(reversed_chunked, chunked.flip(1), atol=1e-6, rtol=1e-5)


def test_v4a2_uses_mlp_not_recurrent_temporal_fusion() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch import nn

    from karting_agent.model.temporal_mlp_state_conditioned import (
        build_temporal_mlp_state_conditioned_model,
    )

    model = build_temporal_mlp_state_conditioned_model(
        pretrained=False,
        frame_stack=5,
        visual_feature_dim=8,
        temporal_hidden_dim=16,
    )

    assert isinstance(model.temporal_fusion, nn.Sequential)
    assert not any(isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)) for module in model.modules())
