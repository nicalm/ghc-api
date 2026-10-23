"""Configuration keeps declaration recovery independent from lossy cleanup."""

import importlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from ghc_api.app import create_app
from ghc_api.generate_config import generate_config_file
from ghc_api.state import State, state


cache_module = importlib.import_module("ghc_api.cache")
main_module = importlib.import_module("ghc_api.main")

FLAG = "enable_portable_schema_recovery"
CLEANUP = "auto_remove_encrypted_content_on_parse_error"


@pytest.fixture
def runtime_client(monkeypatch):
    monkeypatch.setattr(state, FLAG, True)
    monkeypatch.setattr(state, CLEANUP, False)
    monkeypatch.setattr(state, "enable_auth", False)
    return create_app().test_client()


def test_default_enables_declaration_recovery_without_history_cleanup():
    defaults = State()
    assert defaults.enable_portable_schema_recovery is True
    assert defaults.auto_remove_encrypted_content_on_parse_error is False


def test_generated_configuration_exposes_both_independent_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    generate_config_file()
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config[FLAG] is True
    assert config[CLEANUP] is False


def test_runtime_get_exposes_the_declaration_recovery_flag(runtime_client):
    response = runtime_client.get("/api/runtime-config")
    assert response.status_code == 200
    assert response.get_json()[FLAG] is True
    assert response.get_json()[CLEANUP] is False


@pytest.mark.parametrize("cleanup", [False, True])
def test_runtime_toggle_does_not_change_cleanup(runtime_client, monkeypatch, cleanup):
    monkeypatch.setattr(state, CLEANUP, cleanup)
    for enabled in (False, True):
        response = runtime_client.post("/api/runtime-config", json={FLAG: enabled})
        assert response.status_code == 200
        assert response.get_json()["config"][FLAG] is enabled
        assert response.get_json()["config"][CLEANUP] is cleanup
        assert state.enable_portable_schema_recovery is enabled
        assert state.auto_remove_encrypted_content_on_parse_error is cleanup


@pytest.mark.parametrize("invalid", [1, "false", None, [], {}])
def test_runtime_rejects_non_boolean_flags_without_changing_state(runtime_client, invalid):
    response = runtime_client.post("/api/runtime-config", json={FLAG: invalid})
    assert response.status_code == 400
    assert FLAG in response.get_json()["error"]
    assert state.enable_portable_schema_recovery is True
    assert state.auto_remove_encrypted_content_on_parse_error is False


class StartupReached(Exception):
    """Stop a loader test before any app startup, authentication or listening."""


@pytest.fixture
def startup_loader(tmp_path, monkeypatch):
    fresh_state = State()
    monkeypatch.setattr(main_module, "state", fresh_state)
    monkeypatch.setattr(main_module, "get_config_dir", lambda: str(tmp_path))
    monkeypatch.setattr(main_module, "model_mappings", mock.Mock())
    monkeypatch.setattr(main_module, "chat_completions_model_support", mock.Mock())
    monkeypatch.setattr(cache_module, "cache", SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["ghc-api"])
    start = mock.Mock(side_effect=StartupReached)
    monkeypatch.setattr(main_module, "create_app", start)

    def load(settings):
        config = {CLEANUP: False, **settings}
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
        main_module.main()

    return load, fresh_state, start


@pytest.mark.parametrize("settings,expected", [({}, True), ({FLAG: False}, False), ({FLAG: True}, True)])
def test_startup_loads_or_preserves_the_portable_default(startup_loader, settings, expected):
    load, fresh_state, start = startup_loader
    with pytest.raises(StartupReached):
        load(settings)
    assert getattr(fresh_state, FLAG) is expected
    assert getattr(fresh_state, CLEANUP) is False
    start.assert_called_once()


@pytest.mark.parametrize("invalid", [1, "false", None, [], {}])
def test_startup_rejects_non_boolean_portable_config(startup_loader, invalid):
    load, _, start = startup_loader
    with pytest.raises(SystemExit) as stopped:
        load({FLAG: invalid})
    assert stopped.value.code == 1
    start.assert_not_called()
