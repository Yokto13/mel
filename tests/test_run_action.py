def test_imports_work():
    try:
        from run_action import main

        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"
