def test_imports():
    try:
        from baselines.alias_table.one_language_lemma import alias_table_with_lemmas
        from baselines.alias_table.from_tokens import one_language

        from data_processors.tokens.duplicates_filter_script import (
            run_duplicates_filter_script,
        )

        from finetunings.generate_epochs.generate import generate
        from finetunings.finetune_model.train import train
        from finetunings.evaluation.evaluate import evaluate, run_recall_calculation
        from finetunings.file_processing.gathers import (
            move_tokens,
            rename,
            remove_duplicates,
        )
        from tokenization.generate_tokens import (
            tokens_for_finetuning_mewsli,
            tokens_for_finetuning_damuel_descriptions,
            tokens_for_finetuning_damuel_links,
        )

        # If all imports succeed, the test passes
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"
