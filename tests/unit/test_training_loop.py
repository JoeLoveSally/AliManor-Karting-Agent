import pytest

torch = pytest.importorskip("torch")

from karting_agent.train.trainer import run_epoch


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, inputs):
        return self.linear(inputs).squeeze(-1)


def batch():
    return {
        "input": torch.zeros(3, 4),
        "target": torch.tensor([0.0, 1.0, 0.0]),
        "near_transition": torch.tensor([False, True, True]),
        "near_short_correction": torch.tensor([False, False, True]),
    }


def test_run_epoch_reports_subsets_and_updates_model() -> None:
    model = TinyModel()
    before = model.linear.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    summary = run_epoch(
        model,
        [batch()],
        torch.device("cpu"),
        optimizer=optimizer,
    )

    assert summary["all"]["samples"] == 3
    assert summary["transition"]["samples"] == 2
    assert summary["short_correction"]["samples"] == 1
    assert not torch.equal(before, model.linear.weight.detach())


def test_run_epoch_eval_does_not_require_optimizer() -> None:
    model = TinyModel()
    summary = run_epoch(model, [batch()], torch.device("cpu"))

    assert summary["all"]["samples"] == 3
    assert summary["loss"] >= 0.0
