from utils.extractors.extractor_builder import DamuelDescriptionsExtractorBuilder

class Director:
    def construct_descriptions_for_finetuning(self, damuel_path: str, tokenizer, n_of_extractors=None, current_extractor_n=None):
        builder = DamuelDescriptionsExtractorBuilder()
        builder.set_source(damuel_path)
        if n_of_extractors is not None and current_extractor_n is not None:
            assert 0 <= current_extractor_n < n_of_extractors
            builder.set_modulo_file_acceptor(n_of_extractors, current_extractor_n)
