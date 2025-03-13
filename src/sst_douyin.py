# 将抖音的音频文件转换成文字
from typing import Optional
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from fastapi import FastAPI, Depends, HTTPException, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import uvicorn
from datetime import datetime

# 假设这是有效的认证令牌
VALID_TOKEN = "your_valid_token_here"

model_dir = "iic/SenseVoiceSmall"
audio_dir = "data/audio"

app = FastAPI()
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)


def generate_timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S-%f")[:-3]  # 获取毫秒并截取前三位
    return timestamp

def transform_to_text(file_path: str):
    model = AutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
        hub="ms",
    )
    # en
    res = model.generate(
        input=file_path,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    print(text)

    return text

class AudioRequest(BaseModel):
    audio_url: str

class AudioResponse(BaseModel):
    text: str

def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=403, detail="Authorization header missing")
    
    try:
        token_type, token = authorization.split(' ')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid authorization header format")

    if token_type.lower() != "bearer":
        raise HTTPException(status_code=400, detail="Unsupported authorization type")

    if not token or token != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return token

@app.post("/ai-app/extract_text")
async def extract_text(request: AudioRequest, token: str = Depends(verify_token)) -> AudioResponse:
    # 确保 data/audio 文件夹存在
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    try:
        response = requests.get(request.audio_url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to download audio file: {str(e)}")

    audio_file_path = os.path.join(audio_dir, f'{generate_timestamp()}.mp4')
    with open(audio_file_path, 'wb') as f:
        f.write(response.content)

    text = transform_to_text(audio_file_path)
    

    return { "text": text }

def main():
    uvicorn.run('sst_douyin:app', host="0.0.0.0", port=8765, reload=True)


if __name__ == "__main__":
    main()
