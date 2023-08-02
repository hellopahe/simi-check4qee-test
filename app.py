import numpy
import torch
import gradio as gr

from transformers import PegasusForConditionalGeneration, Text2TextGenerationPipeline, AutoModel, AutoTokenizer
from article_extractor.tokenizers_pegasus import PegasusTokenizer
from embed import Embed

import tensorflow as tf

from harvesttext import HarvestText
from sentence_transformers import SentenceTransformer, util
from LexRank import degree_centrality_scores

from luotuo_util import DeviceMap
from peft import get_peft_model, LoraConfig, TaskType


class SummaryExtractor(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PegasusForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese').to(self.device)
        self.tokenizer = PegasusTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")
        self.text2text_genr = Text2TextGenerationPipeline(self.model, self.tokenizer, device=self.device)

    def extract(self, content: str) -> str:
        print(content)
        return str(self.text2text_genr(content, do_sample=False, num_return_sequences=3)[0]["generated_text"])

class Tuoling_6B_extractor(object):
    def __init__(self):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map=DeviceMap("ChatGLM").get())

        # load fine-tuned pretrained model.
        peft_path = "./luotuoC.pt"
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.1)
        self.model = get_peft_model(self.model, peft_config)
        self.model.load_state_dict(torch.load(peft_path), strict=False)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    @staticmethod
    def format_example(example: dict) -> dict:
        context = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            context += f"Input: {example['input']}\n"
        context += "Answer: "
        target = example["output"]
        return {"context": context, "target": target}

    def extract(self, instruction: str, input=None) -> str:
        with torch.no_grad():
            feature = Tuoling_6B_extractor.format_example(
                {"instruction": "请帮我总结以下内容", "output": "", "input": f"{instruction}"}
            )
            input_text = feature["context"]
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            out = self.model.generate(input_ids=input_ids, max_length=2048, temperature=0)
            answer = self.tokenizer.decode(out[0])
            return answer.split('Answer:')[1]

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
        for index in most_central_sentence_indices:
            num -= len(sentences[index])
            if num < 0 and index > 0:
                ptr = index + 1
                break
        return list(sentences[index] for index in most_central_sentence_indices[0: ptr])

# ---===--- worker instances ---===---
t_randeng = SummaryExtractor()
# t_tuoling = Tuoling_6B_extractor()

embedder = Embed()
lex = LexRank()


def randeng_extract(content):
    sentences = lex.find_central(content)
    output = "原文: \n"
    for index, sentence in enumerate(sentences):
        output += f"{index}: {sentence}\n"
    output += "摘要:\n"
    for index, sentence in enumerate(sentences):
        output += f"{index}: {t_randeng.extract(sentence)}\n"
    return output

# def tuoling_extract(content):
#     sentences = lex.find_central(content)
#     return str(list(t_tuoling.extract(sentence) for sentence in sentences))

def similarity_check(query, doc):
    doc_list = doc.split("\n")

    query_embedding = embedder.encode(query)
    doc_embedding = embedder.encode(doc_list)
    scores = (query_embedding @ tf.transpose(doc_embedding))[0].numpy().tolist()
    # scores = list(util.cos_sim(embedding_list[-1], doc_embedding) for doc_embedding in embedding_list[:-1])
    return str(scores)

with gr.Blocks() as app:
    gr.Markdown("从下面的标签选择测试模块 [摘要生成,相似度检测]")
    with gr.Tab("LexRank->Randeng-Pegasus-523M"):
        text_input_1 = gr.Textbox(label="请输入长文本:", max_lines=1000)
        text_output_1 = gr.Textbox(label="摘要文本", lines=10)
        text_button_1 = gr.Button("生成摘要")
    # with gr.Tab("LexRank->Tuoling-6B-chatGLM"):
    #     text_input = gr.Textbox(label="请输入长文本:", max_lines=1000)
    #     text_output = gr.Textbox(label="摘要文本")
    #     text_button = gr.Button("生成摘要")
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
