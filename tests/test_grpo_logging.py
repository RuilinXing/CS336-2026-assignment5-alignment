from scripts.grpo import print_training_metrics


def test_print_training_metrics_includes_required_train_fields(capsys):
    print_training_metrics(
        {
            "step": 42,
            "update_index": 3,
            "split": "train",
            "loss": -0.0132,
            "grad_norm": 0.84,
            "mean_token_entropy": 1.92,
            "mean_reward": 0.31,
            "mean_format_reward": 0.79,
            "clip_fraction": 0.07,
        }
    )

    assert capsys.readouterr().out == (
        "step=42 update=3 loss=-0.0132 grad_norm=0.8400 token_entropy=1.9200 "
        "train_reward=0.3100 train_format_reward=0.7900 clip_fraction=0.0700\n"
    )


def test_print_training_metrics_includes_required_validation_fields(capsys):
    print_training_metrics(
        {
            "step": 50,
            "split": "validation",
            "val_reward": 0.28,
            "val_format_reward": 0.83,
            "val_average_response_length": 217.4,
        }
    )

    assert capsys.readouterr().out == (
        "step=50 val_reward=0.2800 val_format_reward=0.8300 "
        "val_average_response_length=217.4\n"
    )
