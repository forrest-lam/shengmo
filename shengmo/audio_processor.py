"""
音频预处理模块
- 音频格式转换与标准化
- 说话人分离 (pyannote.audio)
- 声纹匹配与过滤
"""

import io
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from .config import ShengMoConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """说话人片段"""
    speaker: str       # 说话人标签, e.g. "SPEAKER_00"
    start: float       # 开始时间 (秒)
    end: float         # 结束时间 (秒)
    is_me: bool = False  # 是否是我的声音


@dataclass
class AudioChunk:
    """音频切片"""
    data: np.ndarray
    sample_rate: int
    segment: SpeakerSegment


class AudioPreprocessor:
    """音频预处理器"""

    def __init__(self, config: ShengMoConfig):
        self.config = config
        self._diarization_pipeline = None
        self._voiceprint_encoder = None
        self._my_embedding = None

    def normalize_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """
        标准化音频：转为 16kHz 单声道 WAV
        """
        audio = AudioSegment.from_file(audio_path)

        # 转为单声道
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # 重采样到目标采样率
        target_sr = self.config.engine.sample_rate
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        # 转为 16bit
        audio = audio.set_sample_width(2)

        # 转为 numpy array
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")

        logger.info(f"音频标准化完成: {len(data)/sr:.1f}s, {sr}Hz, mono")
        return data, sr

    def save_normalized(self, data: np.ndarray, sr: int, output_path: str) -> str:
        """保存标准化后的音频"""
        sf.write(output_path, data, sr)
        return output_path

    # ----------------------------------------------------------------
    # 说话人分离
    # ----------------------------------------------------------------

    def _init_diarization(self):
        """初始化 pyannote 说话人分离模型（懒加载）"""
        if self._diarization_pipeline is not None:
            return

        import os
        import torch
        from pyannote.audio import Pipeline

        hf_token = os.getenv("HF_TOKEN", "")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"加载说话人分离模型: {self.config.speaker_diarization.pyannote_model} (device={device})")
        self._diarization_pipeline = Pipeline.from_pretrained(
            self.config.speaker_diarization.pyannote_model,
            use_auth_token=hf_token,
        )
        self._diarization_pipeline.to(device)

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        """
        对音频进行说话人分离
        返回: 按时间排序的说话人片段列表
        """
        if not self.config.speaker_diarization.enabled:
            # 不分离，整段作为一个 speaker
            data, sr = sf.read(audio_path)
            duration = len(data) / sr
            return [SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=duration)]

        self._init_diarization()

        params = {}
        if self.config.speaker_diarization.max_speakers > 0:
            params["max_speakers"] = self.config.speaker_diarization.max_speakers
        if self.config.speaker_diarization.min_speakers > 0:
            params["min_speakers"] = self.config.speaker_diarization.min_speakers

        logger.info("正在进行说话人分离...")
        diarization = self._diarization_pipeline(audio_path, **params)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            ))

        # 合并同一说话人的相邻片段（间隔 < 0.5s）
        segments = self._merge_adjacent_segments(segments, gap_threshold=0.5)
        logger.info(f"说话人分离完成: {len(segments)} 个片段, "
                     f"{len(set(s.speaker for s in segments))} 位说话人")
        return segments

    @staticmethod
    def _merge_adjacent_segments(segments: list[SpeakerSegment],
                                  gap_threshold: float = 0.5) -> list[SpeakerSegment]:
        """合并同一说话人的相邻片段"""
        if not segments:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg.speaker == prev.speaker and (seg.start - prev.end) < gap_threshold:
                prev.end = seg.end
            else:
                merged.append(seg)
        return merged

    # ----------------------------------------------------------------
    # 声纹匹配
    # ----------------------------------------------------------------

    def _init_voiceprint(self):
        """初始化声纹编码器 (resemblyzer)"""
        if self._voiceprint_encoder is not None:
            return

        from resemblyzer import VoiceEncoder, preprocess_wav

        logger.info("加载声纹编码器...")
        self._voiceprint_encoder = VoiceEncoder()
        self._preprocess_wav = preprocess_wav

        # 计算我的声纹嵌入
        embeddings = []
        for sample_path in self.config.voiceprint.my_voice_samples:
            if not Path(sample_path).exists():
                logger.warning(f"声纹样本不存在: {sample_path}")
                continue
            wav = preprocess_wav(Path(sample_path))
            emb = self._voiceprint_encoder.embed_utterance(wav)
            embeddings.append(emb)

        if embeddings:
            self._my_embedding = np.mean(embeddings, axis=0)
            logger.info(f"声纹注册完成，使用了 {len(embeddings)} 个样本")
        else:
            logger.warning("没有有效的声纹样本，声纹过滤将不可用")
            self._my_embedding = None

    def match_voiceprint(self, audio_data: np.ndarray, sr: int) -> float:
        """
        计算音频与我的声纹的相似度
        返回: 余弦相似度 (0-1)
        """
        if self._my_embedding is None:
            return 1.0

        from resemblyzer import preprocess_wav

        # resemblyzer 需要 int16 格式
        if audio_data.dtype != np.int16:
            if np.max(np.abs(audio_data)) <= 1.0:
                audio_data = (audio_data * 32767).astype(np.int16)

        # 写入临时 buffer 再读取预处理
        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="wav")
        buf.seek(0)
        wav = preprocess_wav(buf)

        emb = self._voiceprint_encoder.embed_utterance(wav)
        similarity = np.dot(emb, self._my_embedding) / (
            np.linalg.norm(emb) * np.linalg.norm(self._my_embedding)
        )
        return float(similarity)

    def filter_my_voice(self, audio_data: np.ndarray, sr: int,
                        segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
        """
        通过声纹匹配标记哪些片段是我的声音
        """
        if not self.config.voiceprint.enabled:
            for seg in segments:
                seg.is_me = True
            return segments

        self._init_voiceprint()

        if self._my_embedding is None:
            for seg in segments:
                seg.is_me = True
            return segments

        threshold = self.config.voiceprint.similarity_threshold

        for seg in segments:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            chunk = audio_data[start_sample:end_sample]

            if len(chunk) < sr * 0.5:  # 片段太短 (<0.5s)，跳过
                seg.is_me = False
                continue

            similarity = self.match_voiceprint(chunk, sr)
            seg.is_me = similarity >= threshold
            logger.debug(f"[{seg.speaker}] {seg.start:.1f}-{seg.end:.1f}s "
                         f"similarity={similarity:.3f} is_me={seg.is_me}")

        my_count = sum(1 for s in segments if s.is_me)
        logger.info(f"声纹过滤: {my_count}/{len(segments)} 个片段匹配为我的声音")
        return segments

    def extract_chunk(self, audio_data: np.ndarray, sr: int,
                      segment: SpeakerSegment) -> AudioChunk:
        """从音频中提取指定片段"""
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        chunk_data = audio_data[start_sample:end_sample]
        return AudioChunk(data=chunk_data, sample_rate=sr, segment=segment)
