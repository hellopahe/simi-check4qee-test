"""
This script summarizes from a paragraph(normally 512 length).
"""

import torch
import gradio as gr

from transformers import PegasusForConditionalGeneration, Text2TextGenerationPipeline
from article_extractor.tokenizers_pegasus import PegasusTokenizer


class SummaryExtractor(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PegasusForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese').to(self.device)
        self.tokenizer = PegasusTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")


        self.text2text_genr = Text2TextGenerationPipeline(self.model, self.tokenizer, device=self.device)

    def extract(self, content: str, min=20, max=30) -> str:
        # inputs = self.tokenizer(content, max_length=512, return_tensors='pt').to(device=self.device)
        # summary_ids = self.model.generate(inputs['input_ids'])
        # return self.tokenizer.batch_decode(summary_ids,
        #                                       skip_special_tokens=True,
        #                                       clean_up_tokenization_spaces=False)[0]
        return str(self.text2text_genr(content, do_sample=False, min_length=min, max_length=max, num_return_sequences=3)[0]["generated_text"])


t_randeng = SummaryExtractor()


def randeng_extract(content):
    return t_randeng.extract(content)

def similarity_check(query: str, doc: str):
    return "similarity result"

with gr.Blocks() as app:
    gr.Markdown("从下面的标签选择不同的摘要模型, 在左侧输入原文")
    # with gr.Tab("CamelBell-Chinese-LoRA"):
    #     text_input = gr.Textbox()
    #     text_output = gr.Textbox()
    #     text_button = gr.Button("生成摘要")
    with gr.Tab("Randeng-Pegasus-523M"):
        text_input_1 = gr.Textbox()
        text_output_1 = gr.Textbox()
        text_button_1 = gr.Button("生成摘要")
    # with gr.Tab("Flip Image"):
    #     with gr.Row():
    #         image_input = gr.Image()
    #         image_output = gr.Image()
    #     image_button = gr.Button("Flip")
    with gr.Tab("相似度检测"):
        with gr.Row():
            text_input_query = gr.Textbox()
            text_input_doc = gr.Textbox()
        text_button_similarity = gr.Button("对比相似度")
        text_output_similarity = gr.Textbox()

    # text_button.click(tuoling_extract, inputs=text_input, outputs=text_output)
    text_button_1.click(randeng_extract, inputs=text_input_1, outputs=text_output_1)
    text_button_similarity.click(similarity_check, inputs=text_input_query, outputs=text_input_doc)

app.launch()
