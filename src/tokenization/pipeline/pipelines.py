from .base import Pipeline

from .loaders import DaMuELDescriptionLoader, DaMuELLinkLoader, MewsliLoader
from .loggers import LoggerStep, StatisticsLogger
from .savers import NPZSaver
from .tokenizers import CuttingTokenizer, SimpleTokenizer


class MewsliMentionPipeline(Pipeline):
    def __init__(
        self,
        mewsli_tsv_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        compress: bool = True,
    ):
        super().__init__()
        self.add(MewsliLoader(mewsli_tsv_path, use_context=False))
        self.add(SimpleTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))


class MewsliContextPipeline(Pipeline):
    def __init__(
        self,
        mewsli_tsv_path: str,
        tokenizer,
        label_token: str,
        expected_size: int,
        output_filename: str,
        compress: bool = True,
    ):
        super().__init__()
        self.add(MewsliLoader(mewsli_tsv_path, use_context=True))
        self.add(CuttingTokenizer(tokenizer, expected_size, label_token))
        self.add(NPZSaver(output_filename, compress))


class DamuelDescriptionMentionPipeline(Pipeline):
    def __init__(
        self,
        damuel_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        require_wiki_page: bool = True,
        compress: bool = True,
        remainder: int = None,
        mod: int = None,
    ):
        super().__init__()
        self.add(
            DaMuELDescriptionLoader(
                damuel_path,
                require_wiki_page=require_wiki_page,
                remainder=remainder,
                mod=mod,
                use_context=False,
            )
        )
        self.add(SimpleTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))


class DamuelDescriptionContextPipeline(Pipeline):
    def __init__(
        self,
        damuel_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        label_token: str,
        compress: bool = True,
        remainder: int = None,
        mod: int = None,
    ):
        super().__init__()
        self.add(
            DaMuELDescriptionLoader(
                damuel_path,
                require_wiki_page=True,
                remainder=remainder,
                mod=mod,
                use_context=True,
                label_token=label_token,
            )
        )
        self.add(SimpleTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))


class DamuelLinkContextPipeline(Pipeline):
    def __init__(
        self,
        damuel_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        label_token: str,
        compress: bool = True,
        remainder: int = None,
        mod: int = None,
        require_link_wiki_origin: bool = True,
    ):
        super().__init__()
        self.add(
            DaMuELLinkLoader(
                path=damuel_path,
                remainder=remainder,
                mod=mod,
                use_context=True,
                require_link_wiki_origin=require_link_wiki_origin,
            )
        )
        self.add(CuttingTokenizer(tokenizer, expected_size, label_token))
        self.add(NPZSaver(output_filename, compress))


class DamuelLinkMentionPipeline(Pipeline):
    def __init__(
        self,
        damuel_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        compress: bool = True,
        remainder: int = None,
        mod: int = None,
        require_link_wiki_origin: bool = True,
        logger: LoggerStep | None = StatisticsLogger(),
    ):
        super().__init__()
        self.add(
            DaMuELLinkLoader(
                path=damuel_path,
                remainder=remainder,
                mod=mod,
                use_context=False,
                require_link_wiki_origin=require_link_wiki_origin,
            )
        )
        if logger is not None:
            self.add(logger)
        self.add(SimpleTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))
