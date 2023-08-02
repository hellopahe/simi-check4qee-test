import numpy
import torch
import gradio as gr

from transformers import PegasusForConditionalGeneration, Text2TextGenerationPipeline
from article_extractor.tokenizers_pegasus import PegasusTokenizer
from embed import Embed

import tensorflow as tf

from harvesttext import HarvestText
from sentence_transformers import SentenceTransformer, util
from LexRank import degree_centrality_scores


class SummaryExtractor(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PegasusForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese').to(self.device)
        self.tokenizer = PegasusTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")
        self.text2text_genr = Text2TextGenerationPipeline(self.model, self.tokenizer, device=self.device)

    def extract(self, content: str) -> str:
        print(content)
        return str(self.text2text_genr(content, min_length=20, do_sample=False, num_return_sequences=3)[0]["generated_text"])

class LexRank(object):
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.ht = HarvestText()
    def find_central(self, content: str):
        sentences = self.ht.cut_sentences(content)
        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        # Compute the pair-wise cosine similarities
        cos_scores = util.cos_sim(embeddings, embeddings).numpy()

        # Compute the centrality for each sentence
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

        # We argsort so that the first element is the sentence with the highest score
        most_central_sentence_indices = numpy.argsort(-centrality_scores)

        num = 500
        ptr = 0
        for index, sentence in enumerate(sentences):
            num -= len(sentence)
            if num < 0 and index > 0:
                ptr = index - 1
                break
            if num < 0 and index == 0:
                ptr = index
                break

        return list(sentences[index] for index in most_central_sentence_indices[0: ptr])

# ---===--- worker instances ---===---
t_randeng = SummaryExtractor()
embedder = Embed()
lex = LexRank()


def randeng_extract(content):
    sentences = lex.find_central(content)

    num = 500
    ptr = 0
    for index, sentence in enumerate(sentences):
        num -= len(sentence)
        if num < 0 and index > 0:
            ptr = index - 1
            break
        if num < 0 and index == 0:
            ptr = index
            break
    print(">>>")
    for ele in sentences[:ptr]:
        print(ele)
    return t_randeng.extract("".join(sentences[:ptr]))


def similarity_check(inputs: list):
    doc_list = inputs[1].split("\n")
    doc_list.append(inputs[0])
    embedding_list = embedder.encode(doc_list)
    scores = (embedding_list[-1] @ tf.transpose(embedding_list[:-1]))[0].numpy().tolist()
    return numpy.array2string(scores, separator=',')

with gr.Blocks() as app:
    gr.Markdown("从下面的标签选择测试模块 [摘要生成,相似度检测]")
    # with gr.Tab("CamelBell-Chinese-LoRA"):
    #     text_input = gr.Textbox()
    #     text_output = gr.Textbox()
    #     text_button = gr.Button("生成摘要")
    with gr.Tab("Randeng-Pegasus-523M"):
        text_input_1 = gr.Textbox(label="请输入长文本:", max_lines=1000)
        text_output_1 = gr.Textbox(label="摘要文本")
        text_button_1 = gr.Button("生成摘要")
    with gr.Tab("相似度检测"):
        with gr.Row():
            text_input_query = gr.Textbox(label="查询文本")
            text_input_doc = gr.Textbox(lines=10, label="逐行输入待比较的文本列表")
        text_button_similarity = gr.Button("对比相似度")
        text_output_similarity = gr.Textbox()

    # text_button.click(tuoling_extract, inputs=text_input, outputs=text_output)
    text_button_1.click(randeng_extract, inputs=text_input_1, outputs=text_output_1)
    text_button_similarity.click(similarity_check, inputs=[text_input_query, text_input_doc], outputs=text_output_similarity)

app.launch(
    # share=True,
    # debug=True
           )
