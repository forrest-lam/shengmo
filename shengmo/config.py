"""
配置管理模块
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv


@dataclass
class EngineConfig:
    model_type: str = "16k_zh_en"
    voice_format: str = "wav"
    sample_rate: int = 16000


@dataclass
class SpeakerDiarizationConfig:
    enabled: bool = True
    backend: str = "pyannote"  # "pyannote" | "tencent"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    max_speakers: int = 0
    min_speakers: int = 1


@dataclass
class VoiceprintConfig:
    enabled: bool = True
    my_voice_samples: list[str] = field(default_factory=list)
    similarity_threshold: float = 0.78


@dataclass
class HotwordItem:
    word: str
    weight: int = 10


@dataclass
class HotwordsConfig:
    enabled: bool = True
    hotword_id: str = ""
    words: list[HotwordItem] = field(default_factory=list)

    def to_hotword_list_str(self) -> str:
        """生成腾讯云 HotwordList 参数格式: '词|权重,词|权重'"""
        if not self.words:
            return ""
        return ",".join(f"{w.word}|{w.weight}" for w in self.words)


@dataclass
class FillerFilterConfig:
    enabled: bool = True
    cloud_filter_level: int = 2
    local_patterns: list[str] = field(default_factory=list)


@dataclass
class CorrectionConfig:
    enabled: bool = True
    use_cloud_oral2written: bool = True
    use_llm_polish: bool = False
    llm_prompt: str = ""


@dataclass
class PronunciationFixConfig:
    enabled: bool = True
    replacements: dict[str, str] = field(default_factory=dict)
    replace_text_id: str = ""


@dataclass
class OutputConfig:
    format: str = "markdown"
    include_timestamps: bool = True
    include_speaker_labels: bool = True
    output_dir: str = "output"


@dataclass
class TencentCloudConfig:
    secret_id: str = ""
    secret_key: str = ""
    appid: str = ""


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"


@dataclass
class ShengMoConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    speaker_diarization: SpeakerDiarizationConfig = field(default_factory=SpeakerDiarizationConfig)
    voiceprint: VoiceprintConfig = field(default_factory=VoiceprintConfig)
    hotwords: HotwordsConfig = field(default_factory=HotwordsConfig)
    filler_filter: FillerFilterConfig = field(default_factory=FillerFilterConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    pronunciation_fix: PronunciationFixConfig = field(default_factory=PronunciationFixConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tencent_cloud: TencentCloudConfig = field(default_factory=TencentCloudConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> ShengMoConfig:
    """加载配置文件和环境变量"""
    # 加载 .env
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)

    # 加载 YAML
    config_file = Path(config_path)
    raw = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    cfg = ShengMoConfig()

    # Engine
    if "engine" in raw:
        e = raw["engine"]
        cfg.engine = EngineConfig(
            model_type=e.get("model_type", cfg.engine.model_type),
            voice_format=e.get("voice_format", cfg.engine.voice_format),
            sample_rate=e.get("sample_rate", cfg.engine.sample_rate),
        )

    # Speaker Diarization
    if "speaker_diarization" in raw:
        sd = raw["speaker_diarization"]
        cfg.speaker_diarization = SpeakerDiarizationConfig(
            enabled=sd.get("enabled", True),
            backend=sd.get("backend", "pyannote"),
            pyannote_model=sd.get("pyannote_model", "pyannote/speaker-diarization-3.1"),
            max_speakers=sd.get("max_speakers", 0),
            min_speakers=sd.get("min_speakers", 1),
        )

    # Voiceprint
    if "voiceprint" in raw:
        vp = raw["voiceprint"]
        cfg.voiceprint = VoiceprintConfig(
            enabled=vp.get("enabled", True),
            my_voice_samples=vp.get("my_voice_samples", []),
            similarity_threshold=vp.get("similarity_threshold", 0.78),
        )

    # Hotwords
    if "hotwords" in raw:
        hw = raw["hotwords"]
        words = []
        for w in hw.get("words", []):
            if isinstance(w, dict):
                words.append(HotwordItem(word=w["word"], weight=w.get("weight", 10)))
            elif isinstance(w, str):
                words.append(HotwordItem(word=w))
        cfg.hotwords = HotwordsConfig(
            enabled=hw.get("enabled", True),
            hotword_id=hw.get("hotword_id", ""),
            words=words,
        )

    # Filler Filter
    if "filler_filter" in raw:
        ff = raw["filler_filter"]
        cfg.filler_filter = FillerFilterConfig(
            enabled=ff.get("enabled", True),
            cloud_filter_level=ff.get("cloud_filter_level", 2),
            local_patterns=ff.get("local_patterns", []),
        )

    # Correction
    if "correction" in raw:
        cr = raw["correction"]
        cfg.correction = CorrectionConfig(
            enabled=cr.get("enabled", True),
            use_cloud_oral2written=cr.get("use_cloud_oral2written", True),
            use_llm_polish=cr.get("use_llm_polish", False),
            llm_prompt=cr.get("llm_prompt", ""),
        )

    # Pronunciation Fix
    if "pronunciation_fix" in raw:
        pf = raw["pronunciation_fix"]
        cfg.pronunciation_fix = PronunciationFixConfig(
            enabled=pf.get("enabled", True),
            replacements=pf.get("replacements", {}),
            replace_text_id=pf.get("replace_text_id", ""),
        )

    # Output
    if "output" in raw:
        o = raw["output"]
        cfg.output = OutputConfig(
            format=o.get("format", "markdown"),
            include_timestamps=o.get("include_timestamps", True),
            include_speaker_labels=o.get("include_speaker_labels", True),
            output_dir=o.get("output_dir", "output"),
        )

    # 腾讯云凭证 (从环境变量)
    cfg.tencent_cloud = TencentCloudConfig(
        secret_id=os.getenv("TENCENT_SECRET_ID", ""),
        secret_key=os.getenv("TENCENT_SECRET_KEY", ""),
        appid=os.getenv("TENCENT_APPID", ""),
    )

    # LLM 配置 (从环境变量)
    cfg.llm = LLMConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )

    return cfg
