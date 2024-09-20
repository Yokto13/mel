from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper
import pytest
from transformers import BertTokenizerFast

from data_processors.tokens.tokens_cutter import TokensCutter, TokensCutterV2
from data_processors.tokens.tokens_cutter import iterate_by_two, iterate_indices


@pytest.fixture
def setup_tokens_cutter():
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-small")
    text = "This is a test text"
    expected_size = 10
    tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size)
    return TokensCutter(text, tokenizer_wrapper, expected_size)


@pytest.fixture
def setup_tokens_cutter_lorem_ipsum():
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-small")
    lorem_ipsum = "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?"
    expected_size = 64
    tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size)
    return TokensCutter(lorem_ipsum, tokenizer_wrapper, expected_size)


def detokenize_text(tokenizer, input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)


class TestTokensCutter:
    def test_cut_mention_name_short_text(self, setup_tokens_cutter):
        name = setup_tokens_cutter.cut_mention_name(slice(0, 4))

        assert name.shape[0] == 10

    def test_cut_mention_name_short_text_decode(self, setup_tokens_cutter):
        name = setup_tokens_cutter.cut_mention_name(slice(0, 4))

        returned_text = detokenize_text(
            setup_tokens_cutter.tokenizer_wrapper.tokenizer, name
        )

        assert returned_text == setup_tokens_cutter.text[0:4]

    @pytest.mark.parametrize(
        "name_slice",
        [
            slice(0, 4),
            slice(0, 10),
            slice(5, 20),
            slice(123, 1423),
            slice(99999, 10**5),
        ],
    )
    def test_cut_mention_name_long_text(self, setup_tokens_cutter, name_slice):
        setup_tokens_cutter.set_text("abcdefghij" * 10000)

        name = setup_tokens_cutter.cut_mention_name(name_slice)

        assert name.shape[0] == 10

    def test_cut_weird_chars_mention_part(self, setup_tokens_cutter):
        weird_char = "\x94"
        setup_tokens_cutter.set_text("a" * 1000 + weird_char + "a" * 1000)

        mention = setup_tokens_cutter.cut_mention_with_context(slice(998, 1005))

        assert mention.shape[0] == 10

    def test_cut_mention_middle_size(self, setup_tokens_cutter_lorem_ipsum):
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(
            slice(200, 220)
        )

        assert mention.shape[0] == 64

    def test_cut_mention_middle_text(self, setup_tokens_cutter_lorem_ipsum):
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(
            slice(200, 220)
        )

        returned_text = detokenize_text(
            setup_tokens_cutter_lorem_ipsum.tokenizer_wrapper.tokenizer, mention
        )

        assert returned_text in setup_tokens_cutter_lorem_ipsum.text
        assert setup_tokens_cutter_lorem_ipsum.text[200:220] in returned_text

    def test_cut_mention_beginning_size(self, setup_tokens_cutter_lorem_ipsum):
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(slice(0, 19))

        assert mention.shape[0] == 64

    def test_cut_mention_beginning_text(self, setup_tokens_cutter_lorem_ipsum):
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(slice(0, 19))

        returned_text = detokenize_text(
            setup_tokens_cutter_lorem_ipsum.tokenizer_wrapper.tokenizer, mention
        )

        assert returned_text in setup_tokens_cutter_lorem_ipsum.text
        assert setup_tokens_cutter_lorem_ipsum.text[0:19] in returned_text

    def test_cut_mention_end_size(self, setup_tokens_cutter_lorem_ipsum):
        text_len = len(setup_tokens_cutter_lorem_ipsum.text)
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(
            slice(text_len - 30, text_len)
        )

        assert mention.shape[0] == 64

    def test_cut_mention_end_text(self, setup_tokens_cutter_lorem_ipsum):
        text_len = len(setup_tokens_cutter_lorem_ipsum.text)
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(
            slice(text_len - 30, text_len)
        )

        returned_text = detokenize_text(
            setup_tokens_cutter_lorem_ipsum.tokenizer_wrapper.tokenizer, mention
        )

        assert returned_text in setup_tokens_cutter_lorem_ipsum.text
        assert (
            setup_tokens_cutter_lorem_ipsum.text[text_len - 30 : text_len]
            in returned_text
        )


class TestTokensCutterV2:
    @pytest.fixture
    def setup_tokens_cutter_v2(self):
        tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-small")
        tokenizer.add_tokens(["[M]"])
        text = (
            "Prague is the capital and largest [M]city[M] of the [M] Czech Republic [M] and the historical capital of [M] Bohemia [M]."
            "[M] Uber large mention which spans more than 10 tokens because we need to test also what happens when mention is larger than expected size [M]"
            "Situated on the [M] Vltava [M] river..."
        )
        expected_size = 10
        return TokensCutterV2(text, tokenizer, expected_size, "[M]")

    @pytest.mark.parametrize("mention_index", [0, 1, 2, 3, 4])
    def test_cut_mention_v2(self, setup_tokens_cutter_v2, mention_index):
        mention_token_id = setup_tokens_cutter_v2.tokenizer.encode(
            "[M]", add_special_tokens=False
        )[0]
        mention = setup_tokens_cutter_v2.cut(mention_index)
        # assert that mention contains mention_token_id twice
        assert (mention == mention_token_id).sum() == 2
        # assert that mention_token_id is not at the start and end of the mention
        assert mention[0] != mention_token_id
        assert mention[-1] != mention_token_id

        assert len(mention) == setup_tokens_cutter_v2.expected_size

    @pytest.mark.parametrize("mention_index", [0, 1, 2, 3, 4])
    def test_cut_mention_v2_special_tokens(self, setup_tokens_cutter_v2, mention_index):
        blabla_tokens = setup_tokens_cutter_v2.tokenizer.encode("blabla")
        # CLS and SEP tokens are added at the beginning and end of the text in our tokenizers
        # We must make sure that cuts do not include these tokens out
        start_token = blabla_tokens[0]
        end_token = blabla_tokens[-1]

        mention = setup_tokens_cutter_v2.cut(mention_index)
        assert mention[0] == start_token
        assert mention[-1] == end_token


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], []),
        ([1], []),
        ([1, 2], [(1, 2)]),
        ([1, 2, 3, 4], [(1, 2), (3, 4)]),
    ],
)
def test_iterate_by_two(iterable, expected):
    assert list(iterate_by_two(iterable)) == expected


@pytest.mark.parametrize(
    "l_start,r_start,max_length,expected",
    [
        (0, 0, 0, []),
        (0, 0, 1, [(0, 0)]),
        (1, 1, 3, [(1, 1), (0, 2)]),
        (2, 2, 4, [(2, 2), (1, 3), (0, None)]),
        (3, 3, 5, [(3, 3), (2, 4), (1, None), (0, None)]),
        (4, 4, 6, [(4, 4), (3, 5), (2, None), (1, None), (0, None)]),
        (0, 1, 2, [(0, 1)]),
        (0, 1, 4, [(0, 1), (None, 2), (None, 3)]),
    ],
)
def test_iterate_indices(l_start, r_start, max_length, expected):
    assert list(iterate_indices(l_start, r_start, max_length)) == expected
