import pytest
from transformers import BertTokenizerFast

from data_processors.tokens.tokens_cutter import TokensCutter


@pytest.fixture
def setup_tokens_cutter():
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-small")
    text = "This is a test text"
    expected_size = 10
    return TokensCutter(text, tokenizer, expected_size)


@pytest.fixture
def setup_tokens_cutter_lorem_ipsum():
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-small")
    lorem_ipsum = "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?"
    expected_size = 64
    return TokensCutter(lorem_ipsum, tokenizer, expected_size)


def detokenize_text(tokenizer, input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)


class TestTokensCutter:
    def test_cut_mention_name_short_text(self, setup_tokens_cutter):
        name = setup_tokens_cutter.cut_mention_name(slice(0, 4))

        assert name["input_ids"].shape[1] == 10

    def test_cut_mention_name_short_text_decode(self, setup_tokens_cutter):
        name = setup_tokens_cutter.cut_mention_name(slice(0, 4))

        returned_text = detokenize_text(
            setup_tokens_cutter.tokenizer, name["input_ids"][0]
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

        assert name["input_ids"].shape[1] == 10

    def test_cut_weird_chars_mention_part(self, setup_tokens_cutter):
        weird_char = "\x94"
        setup_tokens_cutter.set_text("a" * 1000 + weird_char + "a" * 1000)

        mention = setup_tokens_cutter.cut_mention_with_context(slice(998, 1005))

        assert mention["input_ids"].shape[1] == 10

    def test_cut_mention_middle_size(self, setup_tokens_cutter_lorem_ipsum):
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(
            slice(200, 220)
        )

        assert mention["input_ids"].shape[1] == 64

    def test_cut_mention_middle_text(self, setup_tokens_cutter_lorem_ipsum):
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(
            slice(200, 220)
        )

        returned_text = detokenize_text(
            setup_tokens_cutter_lorem_ipsum.tokenizer, mention["input_ids"][0]
        )

        assert returned_text in setup_tokens_cutter_lorem_ipsum.text
        assert setup_tokens_cutter_lorem_ipsum.text[200:220] in returned_text

    def test_cut_mention_beginning_size(self, setup_tokens_cutter_lorem_ipsum):
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(slice(0, 19))

        assert mention["input_ids"].shape[1] == 64

    def test_cut_mention_beginning_text(self, setup_tokens_cutter_lorem_ipsum):
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(slice(0, 19))

        returned_text = detokenize_text(
            setup_tokens_cutter_lorem_ipsum.tokenizer, mention["input_ids"][0]
        )

        assert returned_text in setup_tokens_cutter_lorem_ipsum.text
        assert setup_tokens_cutter_lorem_ipsum.text[0:19] in returned_text

    def test_cut_mention_end_size(self, setup_tokens_cutter_lorem_ipsum):
        text_len = len(setup_tokens_cutter_lorem_ipsum.text)
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(
            slice(text_len - 30, text_len)
        )

        assert mention["input_ids"].shape[1] == 64

    def test_cut_mention_end_text(self, setup_tokens_cutter_lorem_ipsum):
        text_len = len(setup_tokens_cutter_lorem_ipsum.text)
        mention = setup_tokens_cutter_lorem_ipsum.cut_mention_with_context(
            slice(text_len - 30, text_len)
        )

        returned_text = detokenize_text(
            setup_tokens_cutter_lorem_ipsum.tokenizer, mention["input_ids"][0]
        )

        assert returned_text in setup_tokens_cutter_lorem_ipsum.text
        assert (
            setup_tokens_cutter_lorem_ipsum.text[text_len - 30 : text_len]
            in returned_text
        )
