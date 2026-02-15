# tests/test_main.py
from unittest.mock import patch
import builtins
from src import main as main_module

def test_main_calls_train_and_prints(monkeypatch):

    # Patch train so it returns a sample result
    sample_result = {"logreg": {"F1-Score": 0.95, "Accuracy": 0.99}}
    monkeypatch.setattr(main_module, "train", lambda path: sample_result)

    # Patch print to capture output
    printed = []
    def sample_print(*args, **kwargs):
        printed.append(args)

    monkeypatch.setattr(builtins, "print", sample_print)

    # Call main() directly
    main_module.main()

    # Assertions
    assert printed 
    assert any("Final Results:" in str(p) for p in printed)
    assert any(str(sample_result) in str(p) for p in printed)


def test_main_if_name_main(monkeypatch):
    
    sample_result = {"logreg": {"F1-Score": 0.95, "Accuracy": 0.99}}
    monkeypatch.setattr(main_module, "train", lambda path: sample_result)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)

    # Simulate running as __main__
    main_module.__name__ = "__main__"
    main_module.main()
