from app.gesture_engine.action_dispatcher import ActionDispatcher


def test_dispatch_unknown_no_action():
    d = ActionDispatcher(dry_run=True)
    assert d.dispatch("UNKNOWN", 0.9) is None


def test_debounce_prevents_double_fire():
    d = ActionDispatcher(dry_run=True, debounce_ms=1000.0)
    assert d.dispatch("open_palm", 0.9) == "play_pause"
    assert d.dispatch("open_palm", 0.9) is None


def test_update_mapping(tmp_path, monkeypatch):
    from app.gesture_engine import action_dispatcher as ad

    monkeypatch.setattr(ad, "_config_path", lambda: tmp_path / "mappings.json")
    d = ActionDispatcher(dry_run=True)
    d.update_mapping("open_palm", "scroll_up")
    assert d.action_for_gesture("open_palm") == "scroll_up"
