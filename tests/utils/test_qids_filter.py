from unittest.mock import patch
import numpy as np
import pytest

from utils.qid_filter import qid_filter


def mock_remap_qids(qids, _):
    return qids


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_no_filtering_default(mock_qids_remap):
    @qid_filter(1)
    def loader(data, qids):
        return data, qids

    data = np.arange(6).reshape(3, 2)
    qids = np.array([1, 2, 3])
    out_data, out_qids = loader(data, qids)
    assert out_data is data
    assert out_qids is qids


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_invalid_path_raises(mock_qids_remap):
    with pytest.raises(FileNotFoundError):

        @qid_filter(1, filter_path="")
        def loader(data, qids):
            return data, qids

        data = np.arange(6).reshape(3, 2)
        qids = np.array([1, 2, 3])
        _ = loader(data, qids)


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_basic_filtering(mock_qids_remap, tmp_path):
    to_filter = np.array([1, 3])
    filter_file = tmp_path / "filter.npy"
    np.save(filter_file, to_filter)

    @qid_filter(1, filter_path=str(filter_file))
    def loader(data, qids):
        return data, qids

    data = np.array([[10], [20], [30], [40]])
    qids = np.array([1, 2, 3, 4])
    out_data, out_qids = loader(data, qids)
    expected_qids = np.array([2, 4])
    expected_data = np.array([[20], [40]])
    assert np.array_equal(out_qids, expected_qids)
    assert np.array_equal(out_data, expected_data)


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_preserve_dtype(mock_qids_remap, tmp_path):
    to_filter = np.array([2], dtype=np.int64)
    filter_file = tmp_path / "filter.npy"
    np.save(filter_file, to_filter)

    @qid_filter(1, filter_path=str(filter_file))
    def loader(data, qids):
        return data, qids

    qids = np.array([1, 2, 3], dtype=np.int64)
    data = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    out_data, out_qids = loader(data, qids)
    assert out_qids.dtype == qids.dtype
    assert out_data.dtype == data.dtype


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_preserves_function_metadata(mock_qids_remap, tmp_path):
    filter_file = tmp_path / "filter.npy"
    np.save(filter_file, np.array([], dtype=int))

    def dummy_loader(data, qids):
        """Loader docstring."""
        return data, qids

    decorated = qid_filter(1, filter_path=str(filter_file))(dummy_loader)
    assert decorated.__name__ == dummy_loader.__name__
    assert decorated.__doc__ == dummy_loader.__doc__


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_long_tuple(mock_qids_remap, tmp_path):
    to_filter = np.array([1, 3])
    filter_file = tmp_path / "filter.npy"
    np.save(filter_file, to_filter)

    @qid_filter(2, filter_path=str(filter_file))
    def loader(data, qids):
        return data, data, qids, data

    data = np.array([[10], [20], [30], [40]])
    qids = np.array([1, 2, 3, 4])
    out_data, out_data2, out_qids, out_data3 = loader(data, qids)
    expected_qids = np.array([2, 4])
    expected_data = np.array([[20], [40]])
    assert np.array_equal(out_qids, expected_qids)
    assert np.array_equal(out_data, expected_data)
    assert np.array_equal(out_data2, expected_data)
    assert np.array_equal(out_data3, expected_data)


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_no_tuple(mock_qids_remap, tmp_path):
    to_filter = np.array([1, 3])
    filter_file = tmp_path / "filter.npy"
    np.save(filter_file, to_filter)

    @qid_filter(None, filter_path=str(filter_file))
    def loader(qids):
        return qids

    qids = np.array([1, 2, 3, 4])
    out_qids = loader(qids)
    expected_qids = np.array([2, 4])

    assert np.array_equal(out_qids, expected_qids)


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_no_tuple_identity(mock_qids_remap):

    @qid_filter(None)
    def loader(qids):
        return qids

    qids = np.array([1, 2, 3, 4])
    out_qids = loader(qids)
    expected_qids = qids

    assert np.array_equal(out_qids, expected_qids)


@pytest.mark.parametrize("idx", [1, 2])
@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_assert_raises_value_error_idx(mock_qids_remap, idx, tmp_path):
    to_filter = np.array([1, 3])
    filter_file = tmp_path / "filter.npy"
    np.save(filter_file, to_filter)

    @qid_filter(idx, filter_path=str(filter_file))
    def loader(qids):
        return (qids,)

    qids = np.array([1, 2, 3, 4])
    with pytest.raises(IndexError):
        _ = loader(qids)


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_qid_filter_raises_value_error_for_none_index_with_tuple(
    mock_qids_remap, tmp_path
):
    to_filter = np.array([1, 3])
    filter_file = tmp_path / "filter.npy"
    np.save(filter_file, to_filter)

    @qid_filter(None, filter_path=str(filter_file))
    def loader(data, qids):
        return data, qids

    data = np.array([[1, 2], [3, 4]])
    qids = np.array([1, 2])
    with pytest.raises(
        ValueError,
        match="qids_index cannot be None for a tuple result from the decorated function.",
    ):
        loader(data, qids)
