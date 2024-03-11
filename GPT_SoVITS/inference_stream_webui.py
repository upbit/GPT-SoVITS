import os, sys
import re, logging

sys.path.append(os.getcwd())
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

import torch
import tempfile, io, wave
import argparse
from pydub import AudioSegment

from TTS_infer_pack.TTS import TTS, TTS_Config
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = (
    eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 确保直接启动推理UI时也能够设置。
import gradio as gr

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dict_language = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}

cut_method = {
    i18n("不切"): "cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
tts_pipline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path

parser = argparse.ArgumentParser(description="GPT-SoVITS Streaming")
parser.add_argument(
    "-s",
    "--sovits_path",
    type=str,
    default=sovits_path,
    help="SoVITS模型路径",
)
parser.add_argument(
    "-g",
    "--gpt_path",
    type=str,
    default=gpt_path,
    help="GPT模型路径",
)
parser.add_argument(
    "-rw",
    "--ref_wav",
    type=str,
    default="./example/archive_ruanmei_8.wav",
    help="参考音频路径",
)
parser.add_argument(
    "-rt",
    "--prompt_text",
    type=str,
    default="我听不惯现代乐，听戏却极易入迷，琴弦拨动，时间便流往过去。",
    help="参考音频文本",
)
parser.add_argument(
    "-rl",
    "--prompt_language",
    type=str,
    default=i18n("中文"),
    help="参考音频语种",
)

args = parser.parse_args()

SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)

sovits_path = args.sovits_path
gpt_path = args.gpt_path


def get_weights_names():
    SoVITS_names = [sovits_path]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):
            SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [gpt_path]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"):
            GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split("(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {
        "choices": sorted(SoVITS_names, key=custom_sort_key),
        "__type__": "update",
    }, {
        "choices": sorted(GPT_names, key=custom_sort_key),
        "__type__": "update",
    }


SoVITS_names, GPT_names = get_weights_names()

EXAMPLES = [
    [
        i18n("中文"),
        "根据过年的传说，远古时代有一隻凶残年兽，每到岁末就会从海底跑出来吃人。"
        + "人们为了不被年兽吃掉，家家户户都会祭拜祖先祈求平安，也会聚在一起吃一顿丰盛的晚餐。"
        + "后来人们发现年兽害怕红色、噪音与火光，便开始在当天穿上红衣、门上贴上红纸、燃烧爆竹声，藉此把年兽赶走。"
        + "而这些之后也成为过年吃团圆饭、穿红衣、放鞭炮、贴春联的过年习俗。",
    ],
    [
        i18n("中文"),
        "在父母的葬礼上，她穿一身全黑的丧服。她依旧把头发束得很好，如墨的发丝遮掩着她的神情。她没有掉一滴眼泪。"
        + "直到夜色降临，在实验室的屏幕上，数据螺旋制成的层层几何花纹在戏声中变换、舒展、流动，而将那花纹层层剥去后，是她万般呵护、小心制作的秘密。"
        + "阖眼的父亲与母亲，二者冰冷如沉睡般的面容。是辜负。他们没能遵守和外婆的约定。我也没能履行保护父母的承诺，同样辜负了他们。"
        + "唯有科学…不会辜负。她说。少女希望父母平等，便将自己从前的昵称抹去，将名字从二人姓氏中各取一。"
        + "自那以后，每当有人问及家，她的眼中总是闪过一丝迷茫，她似乎把一切都忘了。在这样的状态中，她开始了忽视时间规律的演算。"
        + "由于沉迷科研，她的进食毫不规律，兴致起来便研究几日几夜，到了极限便昏昏睡去。很快，她只借着火萤微弱的光，在每个夜晚收获研究的进展。"
        + "她愈对已有的生命法则置若罔闻，她的前进就愈加迅速；她完全不在乎公式，接着她漠视了生命的意义。"
        + "她只去观察、用双手揣摩，将数据握在手里感受，接着就编纂出新的物种规律。于是，在她的实验室中，那些蕨类植物与花愈发生长地茂盛，"
        + "它们以肉眼可见的速度长高，接着充斥了所有空间。在那花叶开合的缝隙中——笼罩着父母清冷而素净的、由数据汇聚而成的面庞。"
        + "在沉睡的父母将要睁开双眼的时刻——她几乎摧毁了整个星球原本的物种衍变规律，但她仍然在向着自己的目标前进。"
        + "直到她从研究中抬起头，猛烈地望向天空：智识的瞥视降临到了她的身上。",
    ],
    [
        i18n("中文"),
        "神霄折戟录其二"
        + "「嗯，好吃。」被附体的未央变得温柔了许多，也冷淡了很多。"
        + "她拿起弥耳做的馅饼，小口小口吃了起来。第一口被烫到了,还很可爱地吐着舌头吸气。"
        + "「我一下子有点接受不了, 需要消化消化。」用一只眼睛作为代价维持降灵的弥耳自己也拿了一个馅饼，「你再说一 遍?」"
        + "「当年所谓的陨铁其实是神戟。它被凡人折断，铸成魔剑九柄。这一把是雾海魔剑。 加上他们之前已经收集了两柄。「然后你是?」"
        + "「我是曾经的天帝之女，名字已经忘了。我司掌审判与断罪，用你们的话说，就是刑律。」"
        + "因为光禄寺执掌祭祀典礼的事情，所以仪式、祝词什么的，弥耳被老爹逼得倒是能倒背如流。同时因为尽是接触怪力乱神，弥耳也是知道一些小]道的。 神明要是被知道了真正的秘密名讳，就只能任人驱使了。眼前这位未必是忘了。"
        + "「所以朝廷是想重铸神霄之戟吗?」弥耳说服自己接受了这个设定，追问道。"
        + "「我不知道。这具身体的主人并不知道别的事。她只是很愤怒,想要证明自己。」未央把手放在了胸口上。"
        + "「那接下来,我是应该弄个什么送神仪式把你送走吗?」弥耳摸了摸绷带下已经失去功能的眼睛，「然后我的眼睛也会回来?」",
    ],
]


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def stream_inference(
    text,
    text_lang,
    ref_audio_path,
    prompt_text,
    prompt_lang,
    top_k,
    top_p,
    temperature,
    text_split_method,
    batch_size,
    speed_factor,
    ref_text_free,
    split_bucket,
    byte_stream=True,
):
    inputs = {
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size": int(batch_size),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "return_fragment": True,  # IMPORTANT!
    }

    chunks = tts_pipline.run(inputs)
    if byte_stream:
        yield wave_header_chunk()
        for _sr, data in chunks:
            yield data.tobytes()
    else:
        # Send chunk files
        i = 0
        format = "wav"
        for _sr, data in chunks:
            i += 1
            file = f"{tempfile.gettempdir()}/{i}.{format}"
            segment = AudioSegment(data, frame_rate=32000, sample_width=2, channels=1)
            segment.export(file, format=format)
            yield file


with gr.Blocks(title="GPT-SoVITS Streaming Demo") as app:
    gr.Markdown(
        value=i18n(
            "流式输出演示，分句推理后推送到组件中。MacOS下测试通过，Windows下Chrome音频组件请求存在问题，暂时只支持API的流式推理。"
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
        SoVITS_dropdown.change(tts_pipline.init_vits_weights, [SoVITS_dropdown], [])
        GPT_dropdown.change(tts_pipline.init_t2s_weights, [GPT_dropdown], [])

    gr.Markdown(value=i18n("*请上传并填写参考信息"))
    with gr.Row():
        inp_ref = gr.Audio(
            label=i18n("请上传3~10秒内参考音频，超过会报错！"),
            value=args.ref_wav,
            type="filepath",
        )
        with gr.Column():
            ref_text_free = gr.Checkbox(
                label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"),
                value=False,
                interactive=True,
                show_label=True,
            )
            gr.Markdown(i18n("使用无参考文本模式时建议使用微调的GPT"))
            prompt_text = gr.Textbox(
                label=i18n("参考音频的文本"), value=args.prompt_text
            )
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
            value=args.prompt_language,
        )

        def load_text(file):
            with open(file.name, "r", encoding="utf-8") as file:
                return file.read()

        load_button = gr.UploadButton(i18n("加载参考文本"), variant="secondary")
        load_button.upload(load_text, load_button, prompt_text)

    gr.Markdown(
        value=i18n(
            "*请填写需要合成的目标文本。中英混合选中文，日英混合选日文，中日混合暂不支持，非目标语言文本自动遗弃。"
        )
    )
    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(
                label=i18n("需要合成的文本"), value="", lines=9, interactive=True
            )
        with gr.Column():
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
        with gr.Column():
            batch_size = gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                label=i18n("batch_size"),
                value=1,
                interactive=True,
            )
            speed_factor = gr.Slider(
                minimum=0.25,
                maximum=4,
                step=0.05,
                label="speed_factor",
                value=1.0,
                interactive=True,
            )
            split_bucket = gr.Checkbox(
                label=i18n("数据分桶(可能会降低一点计算量,选就对了)"),
                value=True,
                interactive=True,
                show_label=True,
            )
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

    inference_button.click(
        stream_inference,
        [
            text,
            text_language,
            inp_ref,
            prompt_text,
            prompt_language,
            top_k,
            top_p,
            temperature,
            how_to_cut,
            batch_size,
            speed_factor,
            ref_text_free,
            split_bucket,
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
    server_name="127.0.0.1",
    inbrowser=True,
    share=False,
    server_port=8080,
    quiet=True,
)
