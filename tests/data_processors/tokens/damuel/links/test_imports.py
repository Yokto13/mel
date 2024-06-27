import pytest

def test_imports():
    try:
        from data_processors.tokens.damuel.links.both import TokenizerWrapper
        from data_processors.tokens.damuel.links.both import EntryProcessor
        from data_processors.tokens.damuel.links.both import DamuelLinksTokensIteratorBoth
        from data_processors.tokens.damuel.links.for_finetuning import DamuelLinksTokensIteratorFinetuning
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"
