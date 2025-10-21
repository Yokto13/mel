import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output


def process_inputs(pairs):
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

task = (
    "Your task is to determine if the provided Wikipedia description correctly corresponds "
    "to the entity mention found in the query. The entity mention is marked by <M> and </M>. "
    "Check if the description matches the entity. Answer strictly with 'yes' or 'no'.\n"
    "Example:\n"
    "  Query: 'What is the capital of <M>France</M>?'\n"
    "  Description: 'Paris is the capital and largest city of France...'\n"
    "  Answer: no"
    "  Query: 'What is the <M>capital</M> of France?'\n"
    "  Description: 'Paris is the capital and largest city of France...'\n"
    "  Answer: yes"
)

queries = [
    "What is the capital of <M>China</M>?",
    "In order to save Troy, <M>Paris<M> had to be sacrificed.",
    "<M>2. prezident republiky</M> byl zdrcen dohodou z Mnichova.",
] * 2

documents = [
    "Peking (zvuk výslovnost, čínsky v českém přepisu Pej-ťing, pchin-jinem Běijīng, znaky 北京) je hlavní město Čínské lidové republiky. S více než 21 miliony obyvatel je jedním z nejlidnatějších hlavních měst na světě,[2][3] a po Šanghaji druhým nejlidnatějším městem v Číně.",
    "Paris (Ancient Greek: Πάρις, romanized: Páris), also known as Alexander (Ancient Greek: Ἀλέξανδρος, romanized: Aléxandros), is a mythological figure in the story of the Trojan War.",
    "Edvard Beneš (původním jménem Eduard;[pozn. 2] 28. května 1884 Kožlany[4] – 3. září 1948 Sezimovo Ústí) byl československý politik a státník, druhý československý prezident v letech 1935–1948, resp. v letech 1935–1938 a 1945–1948. V období tzv. Druhé republiky (po Mnichovské dohodě ze dne 29. září 1938 do 15. března 1939) a následné německé okupace do května 1945 žil a politicky působil v exilu. Od roku 1940 až do osvobození Československa byl mezinárodně (nejen protihitlerovskou koalicí) uznaným vrcholným představitelem československého odboje a exilovým prezidentem republiky. Úřadujícím československým prezidentem byl opět v letech 1945–1948.",
    "Shanghai[a] is a direct-administered municipality and the most populous urban area in China. The city is located on the Chinese shoreline on the southern estuary of the Yangtze River, with the Huangpu River flowing through it. The population of the city proper is the second largest in the world with around 24.87 million inhabitants in 2023, while the urban area is the most populous in China, with 29.87 million residents.",
    "Paris[a] is the capital and largest city of France, with an estimated city center population of 2,048,472, and a metropolitan population of 13,171,056 as of January 2025[3] in an area of more than 105 km2 (41 sq mi). It is located in the centre of the Île-de-France region. Paris is the fourth-most populous city in the European Union. Nicknamed the City of Light, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy since the 17th century. ",
    "Tomáš Garrigue Masaryk, označovaný T. G. M., TGM nebo Prezident Osvoboditel (7. března 1850 Hodonín[1] – 14. září 1937 Lány[2]), byl československý státník, filozof, sociolog a pedagog, první prezident Československé republiky. K jeho osmdesátým narozeninám byl roku 1930 přijat zákon o zásluhách T. G. Masaryka, obsahující větu „Tomáš Garrigue Masaryk zasloužil se o stát“, a po odchodu z funkce roku 1935 ho parlament znovu ocenil a odměnil za jeho osvoboditelské a budovatelské dílo.",
]

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize the input texts
inputs = process_inputs(pairs)
scores = compute_logits(inputs)

print("scores: ", scores)
