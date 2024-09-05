from unittest.mock import patch, MagicMock

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

    mock_dataset = MagicMock()
    mock_model = MagicMock()
    mock_embs, mock_qids, mock_tokens = MagicMock(), MagicMock(), MagicMock()

    with patch(
        "finetunings.generate_epochs.embed_links_for_generation._get_dataset",
        return_value=mock_dataset,
    ) as mock_get_dataset, patch(
        "finetunings.generate_epochs.embed_links_for_generation._load_model",
        return_value=mock_model,
    ) as mock_load_model, patch(
        "finetunings.generate_epochs.embed_links_for_generation.embed",
        return_value=(mock_embs, mock_qids, mock_tokens),
    ) as mock_embed, patch(
        "finetunings.generate_epochs.embed_links_for_generation._save"
    ) as mock_save:

        embed_links_for_generation(
            links_tokens_dir_path,
            model_path,
            batch_size,
            dest_dir_path,
            state_dict_path,
        )

        mock_get_dataset.assert_called_once_with(links_tokens_dir_path)
        mock_load_model.assert_called_once_with(model_path, state_dict_path)
        mock_embed.assert_called_once_with(
            mock_dataset, mock_model, batch_size, return_qids=True, return_tokens=True
        )
        mock_save.assert_called_once_with(
            dest_dir_path, mock_embs, mock_qids, mock_tokens
        )
