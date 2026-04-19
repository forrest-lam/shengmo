"""
文本后处理模块
- 语气词/停顿词过滤
- 口误/反悔纠正
- 发音纠正（平翘舌、前后鼻音）
- LLM 润色
"""

import re
import logging
from dataclasses import dataclass

from .config import ShengMoConfig
from .asr_engine import RecognitionResult

logger = logging.getLogger(__name__)


class TextPostProcessor:
    """文本后处理器"""

    def __init__(self, config: ShengMoConfig):
        self.config = config
        self._filler_patterns = self._compile_filler_patterns()
        self._pronunciation_map = config.pronunciation_fix.replacements or {}
        self._hotword_fix_map = self._build_hotword_fix_map()

    def _build_hotword_fix_map(self) -> dict[str, str]:
        """
        根据热词配置，构建常见 ASR 误识别 → 正确热词的映射
        英文热词容易被 ASR 拆开或误识别，在此做本地兜底纠正
        """
        fix_map = {}
        if not self.config.hotwords.enabled:
            return fix_map

        # 常见英文热词误识别模式
        # 注意：只纠正"拆词/谐音"类误识别，不要删除热词周围的正确单词
        known_misrecognitions = {
            "deepseek": {
                "Deep seeker", "deep seeker", "Deep Seeker",
                "deep seek", "Deep seek", "DEEP SEEKER", "Deep Seek",
            },
            "kubernetes": {
                "Cooper nettys", "cooper nettys", "Cooper Nettys",
                "cube net ease", "Cube net ease",
            },
            "codebuddy": {
                "code buddy", "Code buddy", "Code Buddy", "code Buddy",
            },
            "langchain": {
                "lang chain", "Lang chain", "Lang Chain", "lang Chain",
            },
        }

        for hw in self.config.hotwords.words:
            word_lower = hw.word.lower()
            if word_lower in known_misrecognitions:
                for wrong in known_misrecognitions[word_lower]:
                    fix_map[wrong] = hw.word

        return fix_map

    def process(self, results: list[RecognitionResult]) -> list[RecognitionResult]:
        """对识别结果进行完整的后处理流水线"""
        processed = []
        for r in results:
            text = r.oral2written_text if r.oral2written_text else r.text

            # 1. 热词纠正（英文热词 ASR 误识别兜底）
            if self.config.hotwords.enabled:
                text = self._fix_hotwords(text)

            # 2. 发音纠正（替换同音错误）
            if self.config.pronunciation_fix.enabled:
                text = self._fix_pronunciation(text)

            # 3. 语气词/停顿词本地过滤（补充云端过滤）
            if self.config.filler_filter.enabled:
                text = self._filter_fillers(text)

            # 4. 口误/反悔纠正（本地规则）
            if self.config.correction.enabled:
                text = self._fix_retraction(text)

            # 5. 清理多余空格和标点
            text = self._clean_text(text)

            processed.append(RecognitionResult(
                text=text,
                speaker=r.speaker,
                start_time=r.start_time,
                end_time=r.end_time,
                words=r.words,
                oral2written_text=r.oral2written_text,
                confidence=r.confidence,
            ))

        # 6. LLM 润色（整体处理）
        if self.config.correction.use_llm_polish:
            processed = self._llm_polish(processed)

        return processed

    # ----------------------------------------------------------------
    # 热词纠正
    # ----------------------------------------------------------------

    def _fix_hotwords(self, text: str) -> str:
        """纠正英文热词的常见 ASR 误识别"""
        # 按 key 长度降序排列，确保长串优先匹配
        for wrong, correct in sorted(self._hotword_fix_map.items(),
                                      key=lambda x: len(x[0]), reverse=True):
            if wrong in text:
                text = text.replace(wrong, correct)
                logger.debug(f"热词纠正: '{wrong}' -> '{correct}'")
        return text

    # ----------------------------------------------------------------
    # 语气词过滤
    # ----------------------------------------------------------------

    def _compile_filler_patterns(self) -> list[re.Pattern]:
        """编译语气词正则"""
        patterns = []
        default_fillers = [
            r"(?:^|(?<=[\u3002\uff01\uff1f\uff0c\u3001，。！？,\.]))\s*那个\s*",
            r"(?:^|(?<=[\u3002\uff01\uff1f\uff0c\u3001，。！？,\.]))\s*就是说\s*",
            r"(?:^|(?<=[\u3002\uff01\uff1f\uff0c\u3001，。！？,\.]))\s*然后呢\s*",
            r"\s*嗯+\s*",
            r"\s*啊+\s*(?=[，。！？,\.\s]|$)",
            r"\s*呃+\s*",
            r"\s*额+\s*",
            r"\s*哦+\s*(?=[，。！？,\.\s]|$)",
            r"(?:^|(?<=[\u3002\uff01\uff1f\uff0c\u3001，。！？,\.]))\s*对对对\s*",
            r"(?:^|(?<=[\u3002\uff01\uff1f\uff0c\u3001，。！？,\.]))\s*是是是\s*",
            r"(?:^|(?<=[\u3002\uff01\uff1f\uff0c\u3001，。！？,\.]))\s*好好好\s*",
        ]

        # 加入用户自定义模式
        for p in self.config.filler_filter.local_patterns:
            default_fillers.append(rf"\s*{p}\s*")

        for p in default_fillers:
            try:
                patterns.append(re.compile(p, re.UNICODE))
            except re.error as e:
                logger.warning(f"无效的语气词正则: {p} -> {e}")

        return patterns

    def _filter_fillers(self, text: str) -> str:
        """过滤语气词"""
        for pattern in self._filler_patterns:
            text = pattern.sub("", text)
        return text.strip()

    # ----------------------------------------------------------------
    # 发音纠正
    # ----------------------------------------------------------------

    def _fix_pronunciation(self, text: str) -> str:
        """
        修正不标准发音导致的识别错误
        利用替换词表 + 拼音相似度
        """
        # 1. 直接替换词表
        for wrong, correct in self._pronunciation_map.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                logger.debug(f"发音纠正: '{wrong}' -> '{correct}'")

        # 2. 拼音相似度纠正（针对常见平翘舌、前后鼻音混淆）
        text = self._pinyin_correction(text)

        return text

    def _pinyin_correction(self, text: str) -> str:
        """
        基于拼音的发音纠正
        处理平翘舌 (z/zh, c/ch, s/sh) 和前后鼻音 (n/ng) 混淆
        """
        try:
            from pypinyin import pinyin, Style
        except ImportError:
            return text

        # 常见的平翘舌/前后鼻音混淆词对
        # 格式: (错误发音词, 正确词, 场景描述)
        confusion_pairs = {
            # 平翘舌
            "只持": "支持",
            "是线": "实现",
            "生么": "什么",
            "住要": "主要",
            "处存": "存储",
            "司考": "思考",
            "知识": "知识",  # 可能被误识为 zisi
            # 前后鼻音
            "经长": "经常",
            "陈功": "成功",
            "分析": "分析",
            "劲头": "劲头",
        }

        for wrong, correct in confusion_pairs.items():
            if wrong in text and wrong not in self._pronunciation_map:
                text = text.replace(wrong, correct)

        return text

    # ----------------------------------------------------------------
    # 口误/反悔纠正
    # ----------------------------------------------------------------

    def _fix_retraction(self, text: str) -> str:
        """
        修正说话人中途反悔/改口的内容
        规则：
        - "不是 X，是 Y" -> Y
        - "X 啊不对 Y" -> Y
        - "X 我说错了 Y" -> Y
        - 连续重复删除
        """
        # 模式1: "不是X，是Y" / "不对，是Y"
        text = re.sub(
            r"不是[^，。！？,\.\s]{1,20}[，,]\s*是([^，。！？,\.\s]{1,20})",
            r"\1",
            text,
        )

        # 模式2: "X 啊不对/不是/我说错了 Y"
        retraction_markers = [
            r"啊不对",
            r"啊不是",
            r"不对不对",
            r"我说错了",
            r"说错了",
            r"不不不",
            r"等等",
        ]
        for marker in retraction_markers:
            pattern = rf"[^，。！？,\.\s]{{1,30}}{marker}[，,]?\s*"
            text = re.sub(pattern, "", text)

        # 模式3: 连续重复删除 (e.g., "我我我觉得" -> "我觉得")
        text = re.sub(r"([\u4e00-\u9fff])\1{2,}", r"\1", text)

        # 模式4: 两字重复 (e.g., "然后然后" -> "然后")
        text = re.sub(r"([\u4e00-\u9fff]{2,4})\1+", r"\1", text)

        return text

    # ----------------------------------------------------------------
    # 文本清理
    # ----------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理多余的空格和标点"""
        # 多个空格 -> 1个
        text = re.sub(r"\s+", " ", text)
        # 多个连续标点 -> 1个
        text = re.sub(r"[，,]{2,}", "，", text)
        text = re.sub(r"[。\.]{2,}", "。", text)
        # 句首标点删除
        text = re.sub(r"^[，,。\.！？]+", "", text)
        return text.strip()

    # ----------------------------------------------------------------
    # LLM 润色
    # ----------------------------------------------------------------

    def _llm_polish(self, results: list[RecognitionResult]) -> list[RecognitionResult]:
        """使用 LLM 对识别结果做进一步润色"""
        if not self.config.llm.api_key:
            logger.warning("未配置 LLM API Key，跳过 LLM 润色")
            return results

        try:
            import requests

            full_text = "\n".join(r.text for r in results)
            prompt = self.config.correction.llm_prompt.replace("{text}", full_text)

            response = requests.post(
                f"{self.config.llm.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.llm.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.llm.model,
                    "messages": [
                        {"role": "system", "content": "你是一个语音转文字的后处理助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                },
                timeout=60,
            )
            response.raise_for_status()
            polished = response.json()["choices"][0]["message"]["content"]

            # 将 LLM 润色后的文本按段落拆分回各个 result
            polished_lines = [l.strip() for l in polished.strip().split("\n") if l.strip()]

            if len(polished_lines) == len(results):
                for i, r in enumerate(results):
                    r.text = polished_lines[i]
            else:
                # 行数不匹配，整体替换到第一个 result
                if results:
                    results[0].text = polished
                    for r in results[1:]:
                        r.text = ""
                    results = [r for r in results if r.text]

            logger.info("LLM 润色完成")

        except Exception as e:
            logger.warning(f"LLM 润色失败: {e}")

        return results
