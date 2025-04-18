from unittest.mock import MagicMock, patch

from finetunings.generate_epochs.embed_links_for_generation import (
    embed_links_for_generation,
)


# horrible amount of mocking...
def test_embed_links_for_generation():
    links_tokens_dir_path = "some/links/tokens/dir/path"
    model_path = "some/model/path"
    batch_size = 32
    dest_dir_path = "some/dest/dir/path"
    state_dict_path = "some/state/dict/path"
    per_save_size = 1000000

    mock_dataset = MagicMock()
    mock_model = MagicMock()
    mock_embs, mock_qids, mock_tokens = MagicMock(), MagicMock(), MagicMock()

    with patch(
        "finetunings.generate_epochs.embed_links_for_generation._get_dataset",
        return_value=mock_dataset,
    ) as mock_get_dataset, patch(
        "finetunings.generate_epochs.embed_links_for_generation.load_model",
        return_value=mock_model,
    ) as mock_load_model, patch(
        "finetunings.generate_epochs.embed_links_for_generation.embed_generator",
        return_value=[(mock_embs, mock_qids, mock_tokens)],
    ) as mock_embed_generator, patch(
        "finetunings.generate_epochs.embed_links_for_generation._save"
    ) as mock_save:

        embed_links_for_generation(
            links_tokens_dir_path,
            model_path,
            batch_size,
            dest_dir_path,
            state_dict_path,
            per_save_size=per_save_size,
        )

        mock_get_dataset.assert_called_once_with(links_tokens_dir_path)
        mock_embed_generator.assert_called_once_with(
            mock_dataset, mock_model, batch_size
        )
