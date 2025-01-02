# -*- coding: utf-8 -*-
import os
import glob
import gradio as gr
import random
import erniebot
import copy
import json
import torch
from detect import ImageClassifier

demo_path = "file"

erniebot.api_type = 'aistudio'
erniebot.access_token = "c932bf4bdf6489c91ed49f0bf026922c0c0bf72d"
models = "ernie-3.5"
amessages = [{'role': 'user',
              'content': "你现在的任务是根据我输入的犬类名称对其进行介绍科普，不要输出markdown语句，仅输出纯文本即可，介绍内容包括但不限于以下四个方面：\n狗类名称：中文名称（英文名称）\n1.起源与历史：介绍这种狗的起源地，以及它们是如何被培育出来的，它们在历史上扮演的角色。\n2.外观特征：描述这种狗的体型、毛色、毛型等外观上的特点。\n3.性格与行为：分析这种狗的性格特点，它们通常的行为习惯，以及它们与人类和其他动物的互动方式。\n4.健康与护理：讨论这种狗常见的健康问题，以及如何进行日常护理，包括饮食、运动和美容等方面。"}]
responses = erniebot.ChatCompletion.create(
    model=models,
    messages=amessages,
)
amessages.append({'role': 'assistant', 'content': responses.result})

classifier = ImageClassifier(model_path='model34.pth', labels_csv='labels.csv', device=torch.device('cpu'))

def clear_input():
    image = None
    advice = None
    return image, advice


def demo_do():
    data_file = glob.glob(os.path.join(demo_path, '*.jpg'))
    image = random.choice(data_file)
    return image


def submit_input(image):
    try:
        global classlabels, device, model, amessages, models
        messages = copy.deepcopy(amessages)
        class_name = classifier.predict(image)
        try:
            with open('database.txt', 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        if class_name in data:
            return (data[class_name])
        else:
            messages.append({'role': 'user',
                             'content': "本次需要你介绍科普的犬类名称为：{},请你按照第一段为总体介绍，后面依次按照以上四个方面进行介绍科普。".format(class_name)})
            advice = erniebot.ChatCompletion.create(
                model=models,
                messages=messages,
            )
            data[class_name] = advice.result
            with open('database.txt', 'w') as f:
                json.dump(data, f)
            return advice.result
    except Exception:
        advice = "出错了，请重试。如果以上操作都不能解决问题，你可能需要联系犬类科普系统的开发者或者查看犬类科普系统的文档，看看是否有关于这个错误的更多信息。"
        return advice


with gr.Blocks(title='犬类科普系统') as forecast:
    gr.Markdown(
        """
        # 欢迎使用!    

        **爱犬如家，温暖随行。**

        忠诚心耿耿，四足踏风尘。守望夜深深，眼中星光沉。                
        """)
    gr.Markdown("请输入您要了解的犬类照片，本系统会**自动识别**并进行**科普讲解**。")
    with gr.Row():
        image = gr.Image(sources=["upload"], label="犬类图片投放处")
        advice = gr.Textbox(label="科普讲解", lines=10, max_lines=40, placeholder="点击“开始生成”即可查看...")
    with gr.Row():
        with gr.Row():
            demo1 = gr.Button("示例展示")
            clear = gr.Button("清空内容")
    submit = gr.Button("开始生成")

    with gr.Accordion("犬类识别系统说明"):
        with gr.Row():
            gr.Markdown(
                """
                ### **犬类科普系统说明**
                
                **本犬类识别系统的科普介绍分为以下内容：**
                
               
                
                - 起源与历史：介绍狗狗的起源地，以及它们是如何被培育出来的，它们在历史上扮演的角色。
                
                - 外观特征：描述狗狗的体型、毛色、毛型等外观上的特点。
                
                - 性格与行为：分析狗狗的性格特点，它们通常的行为习惯，以及它们与人类和其他动物的互动方式。
                
                - 健康与护理：讨论狗狗常见的健康问题，以及如何进行日常护理，包括饮食、运动和美容等方面。             
            
                
                
                **忠胆护家园，足迹绘春秋。守望月华明，夜深人静幽。**

                我们在这里静候您的下次使用！                
                """)

    submit.click(fn=submit_input, inputs=[image], outputs=[advice])
    clear.click(fn=clear_input, inputs=[], outputs=[image,advice])
    demo1.click(fn=demo_do, inputs=[], outputs=[image])
    gr.Markdown("### **本系统由河南理工大学计算机2205班李航训练搭建提供！**")

forecast.launch(share=True)