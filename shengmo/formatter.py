"""
输出格式化模块
支持: text, srt, json, markdown
"""

import json
import logging
from datetime import timedelta
from pathlib import Path

from .asr_engine import RecognitionResult
from .config import ShengMoConfig

logger = logging.getLogger(__name__)


class OutputFormatter:
    """输出格式化器"""

    def __init__(self, config: ShengMoConfig):
        self.config = config

    def format(self, results: list[RecognitionResult], audio_name: str = "") -> str:
        """根据配置格式化输出"""
        fmt = self.config.output.format.lower()
        if fmt == "text":
            return self._format_text(results)
        elif fmt == "srt":
            return self._format_srt(results)
        elif fmt == "json":
            return self._format_json(results)
        elif fmt == "markdown":
            return self._format_markdown(results, audio_name)
        else:
            logger.warning(f"未知输出格式 '{fmt}'，使用 text")
            return self._format_text(results)

    def save(self, results: list[RecognitionResult], output_path: str,
             audio_name: str = "") -> str:
        """格式化并保存到文件"""
        content = self.format(results, audio_name)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info(f"结果已保存: {output_path}")
        return output_path

    # ----------------------------------------------------------------
    # 纯文本
    # ----------------------------------------------------------------

    def _format_text(self, results: list[RecognitionResult]) -> str:
        lines = []
        for r in results:
            if not r.text:
                continue
            prefix = ""
            if self.config.output.include_speaker_labels and r.speaker:
                prefix += f"[{r.speaker}] "
            if self.config.output.include_timestamps and r.start_time:
                ts = self._format_time(r.start_time)
                prefix += f"({ts}) "
            lines.append(f"{prefix}{r.text}")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # SRT 字幕
    # ----------------------------------------------------------------

    def _format_srt(self, results: list[RecognitionResult]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            if not r.text:
                continue
            start = self._format_srt_time(r.start_time)
            end = self._format_srt_time(r.end_time)
            text = r.text
            if self.config.output.include_speaker_labels and r.speaker:
                text = f"[{r.speaker}] {text}"
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # JSON
    # ----------------------------------------------------------------

    def _format_json(self, results: list[RecognitionResult]) -> str:
        data = []
        for r in results:
            if not r.text:
                continue
            item = {"text": r.text}
            if self.config.output.include_speaker_labels and r.speaker:
                item["speaker"] = r.speaker
            if self.config.output.include_timestamps:
                item["start_time"] = r.start_time
                item["end_time"] = r.end_time
            if r.words:
                item["words"] = r.words
            data.append(item)
        return json.dumps(data, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------
    # Markdown
    # ----------------------------------------------------------------

    def _format_markdown(self, results: list[RecognitionResult],
                         audio_name: str = "") -> str:
        lines = []
        lines.append(f"# 语音识别结果")
        if audio_name:
            lines.append(f"\n> 音频文件: `{audio_name}`\n")
        lines.append(f"---\n")

        current_speaker = None
        for r in results:
            if not r.text:
                continue

            # 说话人标题
            if self.config.output.include_speaker_labels and r.speaker:
                if r.speaker != current_speaker:
                    current_speaker = r.speaker
                    lines.append(f"\n### 🎙️ {r.speaker}\n")

            # 时间戳 + 文本
            if self.config.output.include_timestamps and r.start_time:
                ts = self._format_time(r.start_time)
                lines.append(f"**[{ts}]** {r.text}\n")
            else:
                lines.append(f"{r.text}\n")

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------------

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间 -> MM:SS"""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """格式化 SRT 时间 -> HH:MM:SS,mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
