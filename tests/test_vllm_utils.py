from cs336_alignment import vllm_utils


def test_start_server_disables_uvicorn_access_logs(monkeypatch):
    command: list[str] = []

    def fake_popen(args, **kwargs):
        del kwargs
        command.extend(args)
        return object()

    monkeypatch.setattr(vllm_utils.subprocess, "Popen", fake_popen)

    vllm_utils.start_server(
        model_id="local-model",
        host="127.0.0.1",
        port=8000,
        gpu=1,
        seed=0,
        load_format="auto",
        logging_level="ERROR",
    )

    assert "--disable-uvicorn-access-log" in command
