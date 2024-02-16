import os
import tempfile
import gradio as gr
from pydub import AudioSegment
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import (
    get_weights_names,
    custom_sort_key,
    change_choices,
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
    set_seed,
    reuse_seed,
)


i18n = I18nAuto()

gpt_path = os.environ.get(
    "gpt_path",
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
)
sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
SoVITS_names, GPT_names = get_weights_names()

EXAMPLES = [
    [
        "中文",
        "根据过年的传说，远古时代有一隻凶残年兽，每到岁末就会从海底跑出来吃人。"
        + "人们为了不被年兽吃掉，家家户户都会祭拜祖先祈求平安，也会聚在一起吃一顿丰盛的晚餐。"
        + "后来人们发现年兽害怕红色、噪音与火光，便开始在当天穿上红衣、门上贴上红纸、燃烧爆竹声，藉此把年兽赶走。"
        + "而这些之后也成为过年吃团圆饭、穿红衣、放鞭炮、贴春联的过年习俗。",
    ],
    [
        "英文",
        "Give every man thy ear, but few thy voice; Take each man's censure, but reserve thy judgment."
        + " This is a line spoken by 'the tedious old fool' Polonius - chief counsellor to the villain Claudius in Act 1, Scene 3 of Shakespeare's Hamlet."
        + "The implication is that it's important to be a good listener and accept criticism but not be judgmental.",
    ],
]


def get_streaming_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut,
    top_k,
    top_p,
    temperature,
):
    chunks = get_tts_wav(
        ref_wav_path=ref_wav_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        text=text,
        text_language=text_language,
        how_to_cut=how_to_cut,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        stream=True,
    )

    # Send chunk files
    i = 0
    format = "wav"
    for chunk in chunks:
        i += 1
        file = f"{tempfile.gettempdir()}/{i}.{format}"
        segment = AudioSegment(chunk, frame_rate=32000, sample_width=2, channels=1)
        segment.export(file, format=format)
        yield file


def main():
    with gr.Blocks(title="GPT-SoVITS Streaming Demo") as app:
        gr.Markdown(
            value=i18n(
                "流式输出演示，分句推理后推送到组件中。由于目前bytes模式的限制，采用<a href='https://github.com/gradio-app/gradio/blob/gradio%404.17.0/demo/stream_audio_out/run.py'>stream_audio_out</a>中临时文件的方案输出分句。这种方式相比bytes，会增加wav文件解析的延迟。"
            ),
        )

        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(
                label=i18n("GPT模型列表"),
                choices=sorted(GPT_names, key=custom_sort_key),
                value=gpt_path,
                interactive=True,
            )
            SoVITS_dropdown = gr.Dropdown(
                label=i18n("SoVITS模型列表"),
                choices=sorted(SoVITS_names, key=custom_sort_key),
                value=sovits_path,
                interactive=True,
            )
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(
                fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown]
            )
            SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
            GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

        def load_text(file):
            with open(file.name, "r", encoding="utf-8") as file:
                return file.read()

        gr.Markdown(value=i18n("*请上传并填写参考信息"))
        with gr.Row():
            inp_ref = gr.Audio(
                label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath"
            )
            prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="")
            prompt_language = gr.Dropdown(
                label=i18n("参考音频的语种"),
                choices=[
                    i18n("中文"),
                    i18n("英文"),
                    i18n("日文"),
                    i18n("中英混合"),
                    i18n("日英混合"),
                    i18n("多语种混合"),
                ],
                value=i18n("中文"),
            )
            load_button = gr.UploadButton(i18n("加载参考文本"), variant="secondary")
            load_button.upload(load_text, load_button, prompt_text)

        gr.Markdown(
            value=i18n(
                "*请填写需要合成的目标文本。中英混合选中文，日英混合选日文，中日混合暂不支持，非目标语言文本自动遗弃。"
            )
        )
        with gr.Row():
            text = gr.Textbox(
                label=i18n("需要合成的文本"), value="", lines=5, interactive=True
            )
            text_language = gr.Dropdown(
                label=i18n("需要合成的语种"),
                choices=[
                    i18n("中文"),
                    i18n("英文"),
                    i18n("日文"),
                    i18n("中英混合"),
                    i18n("日英混合"),
                    i18n("多语种混合"),
                ],
                value=i18n("中文"),
            )
            how_to_cut = gr.Radio(
                label=i18n("怎么切"),
                choices=[
                    i18n("不切"),
                    i18n("凑四句一切"),
                    i18n("凑50字一切"),
                    i18n("按中文句号。切"),
                    i18n("按英文句号.切"),
                    i18n("按标点符号切"),
                ],
                value=i18n("按标点符号切"),
                interactive=True,
            )

        gr.Markdown(value=i18n("* 参数设置"))
        with gr.Row():
            with gr.Column():
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    label=i18n("top_k"),
                    value=5,
                    interactive=True,
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label=i18n("top_p"),
                    value=1,
                    interactive=True,
                )
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label=i18n("temperature"),
                    value=1,
                    interactive=True,
                )
            with gr.Row():
                seed = gr.Number(label=i18n("种子"), value=-1, precision=0)
                last_seed = gr.State(value=-1)
                reuse_button = gr.Button(value=i18n("种子复用"))
            inference_button = gr.Button(i18n("合成语音"), variant="primary")

        gr.Markdown(value=i18n("* 结果输出(等待第2句推理结束后会自动播放)"))
        with gr.Row():
            audio_file = gr.Audio(
                value=None,
                label=i18n("输出的语音"),
                streaming=True,
                autoplay=True,
                interactive=False,
                show_label=True,
            )

        reuse_button.click(reuse_seed, last_seed, seed)
        inference_button.click(set_seed, seed, last_seed).then(
            get_streaming_tts_wav,
            [
                inp_ref,
                prompt_text,
                prompt_language,
                text,
                text_language,
                how_to_cut,
                top_k,
                top_p,
                temperature,
            ],
            [audio_file],
        ).then(lambda: gr.update(interactive=True), None, [text], queue=False)

        with gr.Row():
            gr.Examples(
                EXAMPLES,
                [text_language, text],
                cache_examples=False,
                run_on_click=False,  # Will not work , user should submit it
            )

    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=8080,
        quiet=True,
    )


if __name__ == "__main__":
    main()
