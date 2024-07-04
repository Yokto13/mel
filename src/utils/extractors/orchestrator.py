from functools import partial
from multiprocessing import Pool

import numpy as np
from transformers import BertTokenizerFast

from utils.extractors.damuel import DamuelExtractor
from utils.extractors.damuel.descriptions import DamuelDescriptionsIterator
from utils.argument_wrappers import ensure_datatypes

class Orchestrator:
    def __init__(self, partialized_constructor, n_of_processes, output_dir) -> None:
        self.iterator_constructor = partialized_constructor
        self.n_of_processes = n_of_processes
        self.output_dir = output_dir

    def run(self):
        iterators = self._get_iterators()
        output_dirs = [f"{self.output_dir}_{i}" for i in range(self.n_of_processes)]
        with Pool(self.n_of_processes) as p:
            p.starmap(iterate, zip(iterators, output_dirs))

    def _get_choosers(self):
        return [partial(is_k_th, n=self.n_of_processes, k=i)
            for i in range(self.n_of_processes)]

    def _get_iterators(self):
        return [self.iterator_constructor(chooser)
            for chooser in self._get_choosers()]

def is_k_th(x, n, k):
    return x % n == k


def iterate(iterator: DamuelExtractor, output_path: str):
    data = []
    print("Starting iteration")
    i = 0
    for x in iterator:
        data.append(x)
        if len(data) > 100000:
            data = np.array(data)
            save(data, f"{output_path}_{i}.npz")
            data = []
            i+=1
    print("Finished iteration")

    # TODO: think whether there isn't better way to do this
    data = np.array(data) 

    print("Saving data")
    save(data, f"{output_path}_{i}.npz")
    print("Data saved")


def save(data, output_dir):
    np.savez_compressed(output_dir, data=data)


@ensure_datatypes(
        [
    int,
    str,
    str,
    str,
    str,
    int,], {}
)
def damuel_description_tokens(
    n_of_processes,
    damuel_path,
    output_dir,
    tokenizer_name,
    name_token="[M]",
    expected_size=64,
):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    tokenizer.add_tokens([name_token])

    constructor = partial(DamuelDescriptionsIterator, damuel_path, tokenizer, name_token, expected_size)

    orchestrator = Orchestrator(constructor, n_of_processes, output_dir)

    orchestrator.run()