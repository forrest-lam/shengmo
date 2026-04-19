"""
腾讯云 ASR 识别模块
支持：
- 录音文件识别 (CreateRecTask) - 支持说话人分离、口语转书面语
- 一句话识别 (SentenceRecognition) - 适合短音频
- 实时语音识别 (WebSocket) - 边说边出
"""

import base64
import json
import io
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from .config import ShengMoConfig

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """识别结果"""
    text: str                          # 识别文本
    speaker: str = ""                  # 说话人标签
    start_time: float = 0.0           # 开始时间
    end_time: float = 0.0             # 结束时间
    words: list[dict] = field(default_factory=list)  # 词级别详情
    oral2written_text: str = ""       # 口语转书面语结果
    confidence: float = 0.0


class TencentASR:
    """腾讯云 ASR 封装"""

    def __init__(self, config: ShengMoConfig):
        self.config = config
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化腾讯云 SDK 客户端"""
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.asr.v20190614 import asr_client

        cred = credential.Credential(
            self.config.tencent_cloud.secret_id,
            self.config.tencent_cloud.secret_key,
        )
        http_profile = HttpProfile()
        http_profile.endpoint = "asr.tencentcloudapi.com"
        http_profile.reqMethod = "POST"

        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client_profile.signMethod = "HmacSHA256"

        self._client = asr_client.AsrClient(cred, "", client_profile)
        logger.info("腾讯云 ASR 客户端初始化完成")

    # ----------------------------------------------------------------
    # 一句话识别 (≤60s)
    # ----------------------------------------------------------------

    def recognize_sentence(self, audio_data: np.ndarray, sr: int) -> RecognitionResult:
        """
        一句话识别，适合 60 秒以内的短音频
        """
        from tencentcloud.asr.v20190614 import models

        # numpy -> wav bytes -> base64
        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="wav")
        wav_bytes = buf.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        req = models.SentenceRecognitionRequest()
        params = {
            "EngSerViceType": self.config.engine.model_type,
            "SourceType": 1,
            "VoiceFormat": "wav",
            "Data": audio_b64,
            "DataLen": len(wav_bytes),
            "WordInfo": 2,  # 含标点的词级别时间戳
        }

        # 热词
        if self.config.hotwords.enabled:
            hotword_str = self.config.hotwords.to_hotword_list_str()
            if hotword_str:
                params["HotwordList"] = hotword_str
            if self.config.hotwords.hotword_id:
                params["HotwordId"] = self.config.hotwords.hotword_id

        # 语气词过滤
        if self.config.filler_filter.enabled:
            params["FilterModal"] = self.config.filler_filter.cloud_filter_level

        # 发音纠正 - 替换词表
        if self.config.pronunciation_fix.replace_text_id:
            params["ReplaceTextId"] = self.config.pronunciation_fix.replace_text_id

        req.from_json_string(json.dumps(params))

        try:
            resp = self._client.SentenceRecognition(req)
            result = json.loads(resp.to_json_string())

            words = []
            if "WordList" in result:
                for w in result["WordList"]:
                    words.append({
                        "word": w.get("Word", ""),
                        "start_time": w.get("StartTime", 0),
                        "end_time": w.get("EndTime", 0),
                    })

            return RecognitionResult(
                text=result.get("Result", ""),
                words=words,
            )
        except Exception as e:
            logger.error(f"一句话识别失败: {e}")
            raise

    # ----------------------------------------------------------------
    # 录音文件识别 (支持说话人分离、口语转书面语)
    # ----------------------------------------------------------------

    def recognize_file(self, audio_path: str,
                       enable_speaker_diarization: bool = False) -> list[RecognitionResult]:
        """
        录音文件识别 - 异步接口，支持长音频
        """
        task_id = self._create_rec_task(audio_path, enable_speaker_diarization)
        return self._poll_rec_task(task_id)

    def _create_rec_task(self, audio_path: str,
                         enable_speaker_diarization: bool) -> int:
        """创建录音文件识别任务"""
        from tencentcloud.asr.v20190614 import models

        # 读取音频 base64
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        req = models.CreateRecTaskRequest()

        # 根据是否启用口语转书面语决定 ResTextFormat
        res_format = 2  # 默认含标点的词粒度
        if self.config.correction.use_cloud_oral2written:
            res_format = 5  # 口语转书面语（增值）

        params = {
            "EngineModelType": self.config.engine.model_type,
            "ChannelNum": 1,
            "ResTextFormat": res_format,
            "SourceType": 1,
            "Data": audio_b64,
            "DataLen": len(audio_bytes),
        }

        # 说话人分离
        if enable_speaker_diarization:
            params["SpeakerDiarization"] = 1
            params["SpeakerNumber"] = self.config.speaker_diarization.max_speakers

        # 热词
        if self.config.hotwords.enabled:
            hotword_str = self.config.hotwords.to_hotword_list_str()
            if hotword_str:
                params["HotwordList"] = hotword_str
            if self.config.hotwords.hotword_id:
                params["HotwordId"] = self.config.hotwords.hotword_id

        # 语气词过滤
        if self.config.filler_filter.enabled:
            params["FilterModal"] = self.config.filler_filter.cloud_filter_level

        # 脏词过滤
        params["FilterDirty"] = 1

        # 发音纠正 - 替换词表
        if self.config.pronunciation_fix.replace_text_id:
            params["ReplaceTextId"] = self.config.pronunciation_fix.replace_text_id

        req.from_json_string(json.dumps(params))

        try:
            resp = self._client.CreateRecTask(req)
            result = json.loads(resp.to_json_string())
            task_id = result["Data"]["TaskId"]
            logger.info(f"录音文件识别任务已创建: TaskId={task_id}")
            return task_id
        except Exception as e:
            logger.error(f"创建录音文件识别任务失败: {e}")
            raise

    def _poll_rec_task(self, task_id: int,
                       max_wait: int = 600, interval: int = 3) -> list[RecognitionResult]:
        """轮询等待识别任务完成"""
        from tencentcloud.asr.v20190614 import models

        req = models.DescribeTaskStatusRequest()
        req.from_json_string(json.dumps({"TaskId": task_id}))

        elapsed = 0
        while elapsed < max_wait:
            try:
                resp = self._client.DescribeTaskStatus(req)
                result = json.loads(resp.to_json_string())
                data = result.get("Data", {})
                status = data.get("StatusStr", "")

                if status == "success":
                    return self._parse_rec_result(data)
                elif status == "failed":
                    err_msg = data.get("ErrorMsg", "unknown error")
                    raise RuntimeError(f"识别任务失败: {err_msg}")
                else:
                    logger.debug(f"任务进行中 ({elapsed}s)...")
                    time.sleep(interval)
                    elapsed += interval
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(f"查询任务状态失败: {e}")
                time.sleep(interval)
                elapsed += interval

        raise TimeoutError(f"识别任务超时 ({max_wait}s)")

    @staticmethod
    def _parse_rec_result(data: dict) -> list[RecognitionResult]:
        """解析录音文件识别结果"""
        results = []
        result_detail = data.get("ResultDetail", [])

        if result_detail:
            # 有详细结果（分段）
            for seg in result_detail:
                words = []
                for w in seg.get("Words", []):
                    words.append({
                        "word": w.get("Word", ""),
                        "start_time": w.get("OffsetStartMs", 0),
                        "end_time": w.get("OffsetEndMs", 0),
                    })

                results.append(RecognitionResult(
                    text=seg.get("FinalSentence", ""),
                    speaker=seg.get("SpeakerId", ""),
                    start_time=seg.get("StartMs", 0) / 1000.0,
                    end_time=seg.get("EndMs", 0) / 1000.0,
                    words=words,
                    oral2written_text=seg.get("WrittenText", ""),
                ))
        else:
            # 只有基础结果
            text = data.get("Result", "")
            results.append(RecognitionResult(text=text))

        return results

    # ----------------------------------------------------------------
    # 实时语音识别 (WebSocket)
    # ----------------------------------------------------------------

    def create_realtime_recognizer(self, callback=None):
        """
        创建实时语音识别器（WebSocket 流式）
        callback: 回调函数, 签名 callback(text: str, is_final: bool, index: int)
        """
        return RealtimeRecognizer(self.config, callback)


class RealtimeRecognizer:
    """实时语音识别（WebSocket 方式）"""

    def __init__(self, config: ShengMoConfig, callback=None):
        self.config = config
        self.callback = callback
        self._ws = None
        self._results: list[RecognitionResult] = []

    def start(self):
        """开始实时识别"""
        import hashlib
        import hmac
        import urllib.parse
        import websocket

        appid = self.config.tencent_cloud.appid
        secret_id = self.config.tencent_cloud.secret_id
        secret_key = self.config.tencent_cloud.secret_key
        engine = self.config.engine.model_type
        timestamp = int(time.time())
        expired = timestamp + 86400

        # 构建签名
        import uuid
        voice_id = str(uuid.uuid4())
        nonce = np.random.randint(1, 2**31 - 1)

        params = {
            "secretid": secret_id,
            "timestamp": str(timestamp),
            "expired": str(expired),
            "nonce": str(nonce),
            "engine_model_type": engine,
            "voice_id": voice_id,
            "voice_format": 1,  # pcm
            "needvad": 1,
            "word_info": 2,
        }

        # 热词
        if self.config.hotwords.enabled:
            hotword_str = self.config.hotwords.to_hotword_list_str()
            if hotword_str:
                params["hotword_list"] = hotword_str

        # 语气词过滤
        if self.config.filler_filter.enabled:
            params["filter_modal"] = str(self.config.filler_filter.cloud_filter_level)

        # 签名
        sign_str = "asr.cloud.tencent.com/asr/v2/" + appid + "?"
        sorted_params = sorted(params.items())
        sign_str += "&".join(f"{k}={v}" for k, v in sorted_params)

        signature = base64.b64encode(
            hmac.new(secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
        ).decode()
        params["signature"] = urllib.parse.quote(signature)

        url = f"wss://asr.cloud.tencent.com/asr/v2/{appid}?" + \
              "&".join(f"{k}={v}" for k, v in params.items())

        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        logger.info("实时语音识别已启动")

    def send_audio(self, audio_chunk: bytes):
        """发送音频数据"""
        if self._ws:
            self._ws.send(audio_chunk, opcode=0x2)

    def stop(self):
        """停止识别"""
        if self._ws:
            # 发送结束标志
            self._ws.send('{"type": "end"}')
            self._ws.close()

    def _on_open(self, ws):
        logger.info("WebSocket 连接已建立")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            code = data.get("code", -1)
            if code != 0:
                logger.warning(f"识别错误: {data.get('message', '')}")
                return

            result = data.get("result", {})
            text = result.get("voice_text_str", "")
            slice_type = result.get("slice_type", 0)
            index = result.get("index", 0)
            is_final = slice_type == 2

            if text and self.callback:
                self.callback(text, is_final, index)

            if is_final:
                self._results.append(RecognitionResult(
                    text=text,
                    start_time=result.get("start_time", 0) / 1000.0,
                    end_time=result.get("end_time", 0) / 1000.0,
                ))
        except Exception as e:
            logger.error(f"解析识别结果失败: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket 错误: {error}")

    def _on_close(self, ws, code, reason):
        logger.info(f"WebSocket 已关闭: {code} {reason}")

    @property
    def results(self) -> list[RecognitionResult]:
        return self._results
