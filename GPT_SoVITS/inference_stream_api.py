# Usage: uvicorn GPT_SoVITS.inference_stream_api:app --host 0.0.0.0 --port 5000
import os, sys

current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "GPT_SoVITS"))

import logging

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
from typing import Annotated
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse, Response
from pydub import AudioSegment

from TTS_infer_pack.TTS import TTS, TTS_Config
from tools.i18n.i18n import I18nAuto

app = FastAPI()
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

EXAMPLES = [
    [
        "中文",
        "根据过年的传说，远古时代有一隻凶残年兽，每到岁末就会从海底跑出来吃人。"
        + "人们为了不被年兽吃掉，家家户户都会祭拜祖先祈求平安，也会聚在一起吃一顿丰盛的晚餐。"
        + "后来人们发现年兽害怕红色、噪音与火光，便开始在当天穿上红衣、门上贴上红纸、燃烧爆竹声，藉此把年兽赶走。"
        + "而这些之后也成为过年吃团圆饭、穿红衣、放鞭炮、贴春联的过年习俗。",
    ],
    [
        "中文",
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
        "中文",
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


@app.get("/tts")
async def tts(
    text: str,  # 必选参数
    language: str = i18n("中文"),
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    range: Annotated[str, Header()] = None,
):
    # TODO: change by args
    ref_audio_path = "./example/archive_ruanmei_8.wav"
    ref_prompt_text = "我听不惯现代乐，听戏却极易入迷，琴弦拨动，时间便流往过去。"
    ref_prompt_lang = i18n("中文")
    how_to_cut = i18n("按标点符号切")
    print(range)

    if range == None:  # Chrome fetch and then get media with range="bytes=0-"
        # fetch
        return Response(content=wave_header_chunk(), media_type="audio/x-wav")
    else:
        # media
        return StreamingResponse(
            stream_inference(
                text=text,
                text_lang=language,
                ref_audio_path=ref_audio_path,
                prompt_text=ref_prompt_text,
                prompt_lang=ref_prompt_lang,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                text_split_method=how_to_cut,
                batch_size=1,
                speed_factor=1.0,
                ref_text_free=False,
                split_bucket=True,
                byte_stream=True,
            ),
            media_type="audio/x-wav",
        )
