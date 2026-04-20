"""
声墨 (ShengMo) 核心 Pipeline
将音频预处理、ASR 识别、文本后处理串联为完整流水线
"""

import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from .config import ShengMoConfig, load_config
from .audio_processor import AudioPreprocessor, SpeakerSegment
from .asr_engine import TencentASR, RecognitionResult
from .text_processor import TextPostProcessor
from .formatter import OutputFormatter

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Pipeline 处理结果"""
    results: list[RecognitionResult] = field(default_factory=list)
    speaker_segments: list[SpeakerSegment] = field(default_factory=list)
    formatted_output: str = ""
    output_path: str = ""
    audio_duration: float = 0.0
    speakers_count: int = 0
    my_segments_count: int = 0


class ShengMoPipeline:
    """
    声墨 ASR Pipeline

    处理流程:
    1. 音频标准化 (格式/采样率/声道)
    2. 说话人分离 (pyannote.audio / 腾讯云)
    3. 声纹匹配过滤 (只保留我的声音)
    4. ASR 语音识别 (腾讯云)
    5. 文本后处理 (语气词/口误/发音纠正)
    6. 格式化输出
    """

    def __init__(self, config: ShengMoConfig | None = None,
                 config_path: str = "config.yaml"):
        if config is None:
            config = load_config(config_path)
        self.config = config
        self.audio_processor = AudioPreprocessor(config)
        self.asr = TencentASR(config)
        self.text_processor = TextPostProcessor(config)
        self.formatter = OutputFormatter(config)

    def process(self, audio_path: str,
                only_my_voice: bool = True,
                output_path: str | None = None) -> PipelineResult:
        """
        处理单个音频文件

        Args:
            audio_path: 音频文件路径
            only_my_voice: 是否只保留我的声音
            output_path: 输出文件路径 (可选, 自动生成)

        Returns:
            PipelineResult 包含所有处理结果
        """
        audio_name = Path(audio_path).stem
        pipeline_result = PipelineResult()

        # ============================================================
        # Step 1: 音频标准化
        # ============================================================
        logger.info(f"📢 开始处理: {audio_path}")
        audio_data, sr = self.audio_processor.normalize_audio(audio_path)
        pipeline_result.audio_duration = len(audio_data) / sr
        logger.info(f"  音频时长: {pipeline_result.audio_duration:.1f}s")

        # 保存标准化音频到临时文件
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, audio_data, sr)
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()

        # ============================================================
        # Step 2: 说话人分离
        # ============================================================
        if self.config.speaker_diarization.enabled and \
           self.config.speaker_diarization.backend == "pyannote":
            segments = self.audio_processor.diarize(tmp_wav_path)
        else:
            # 不做本地分离，交给腾讯云处理
            segments = [SpeakerSegment(
                speaker="SPEAKER_00",
                start=0.0,
                end=pipeline_result.audio_duration,
            )]

        pipeline_result.speaker_segments = segments
        pipeline_result.speakers_count = len(set(s.speaker for s in segments))

        # ============================================================
        # Step 3: 声纹匹配 (只保留我的声音)
        # ============================================================
        if only_my_voice and self.config.voiceprint.enabled:
            segments = self.audio_processor.filter_my_voice(audio_data, sr, segments)
            my_segments = [s for s in segments if s.is_me]
            pipeline_result.my_segments_count = len(my_segments)
            logger.info(f"  我的片段: {len(my_segments)}/{len(segments)}")
        else:
            my_segments = segments
            for s in my_segments:
                s.is_me = True
            pipeline_result.my_segments_count = len(my_segments)

        # ============================================================
        # Step 4: ASR 一句话识别 (每片段 ≤60s)
        # ============================================================
        all_results = []

        if not my_segments:
            logger.warning("没有匹配到我的语音片段")
        else:
            logger.info(f"  使用一句话识别, 共 {len(my_segments)} 个片段...")
            for i, seg in enumerate(my_segments):
                # 如果片段超过 60s，需要再切分
                sub_segments = self._split_segment_if_needed(seg, max_duration=55.0)
                for sub_seg in sub_segments:
                    chunk = self.audio_processor.extract_chunk(audio_data, sr, sub_seg)
                    try:
                        result = self.asr.recognize_sentence(chunk.data, chunk.sample_rate)
                        result.speaker = sub_seg.speaker
                        result.start_time = sub_seg.start
                        result.end_time = sub_seg.end
                        all_results.append(result)
                    except Exception as e:
                        logger.warning(f"  片段识别失败 [{sub_seg.start:.1f}-{sub_seg.end:.1f}s]: {e}")

        # ============================================================
        # Step 5: 文本后处理
        # ============================================================
        logger.info("  文本后处理...")
        all_results = self.text_processor.process(all_results)
        pipeline_result.results = all_results

        # ============================================================
        # Step 6: 格式化输出
        # ============================================================
        pipeline_result.formatted_output = self.formatter.format(all_results, audio_name)

        if output_path is None:
            ext_map = {"text": ".txt", "srt": ".srt", "json": ".json", "markdown": ".md"}
            ext = ext_map.get(self.config.output.format, ".txt")
            output_dir = Path(self.config.output.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{audio_name}{ext}")

        self.formatter.save(all_results, output_path, audio_name)
        pipeline_result.output_path = output_path

        # 清理临时文件
        Path(tmp_wav_path).unlink(missing_ok=True)

        logger.info(f"✅ 处理完成: {output_path}")
        return pipeline_result

    @staticmethod
    def _split_segment_if_needed(seg: SpeakerSegment,
                                  max_duration: float = 55.0) -> list[SpeakerSegment]:
        """将超过 max_duration 的片段切分为多个子片段"""
        duration = seg.end - seg.start
        if duration <= max_duration:
            return [seg]

        sub_segments = []
        start = seg.start
        while start < seg.end:
            end = min(start + max_duration, seg.end)
            sub_segments.append(SpeakerSegment(
                speaker=seg.speaker,
                start=start,
                end=end,
                is_me=seg.is_me,
            ))
            start = end
        return sub_segments

    # ----------------------------------------------------------------
    # 声纹注册
    # ----------------------------------------------------------------

    def register_voiceprint(self, audio_paths: list[str]) -> bool:
        """
        注册我的声纹
        提供几段我说话的音频，系统会提取声纹特征用于后续过滤
        """
        self.config.voiceprint.my_voice_samples = audio_paths
        self.config.voiceprint.enabled = True

        # 重新初始化声纹模块
        self.audio_processor._voiceprint_encoder = None
        self.audio_processor._my_embedding = None
        self.audio_processor._init_voiceprint()

        return self.audio_processor._my_embedding is not None
