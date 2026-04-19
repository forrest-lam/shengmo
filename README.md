# 🎙️ 声沫 (ShengMo) - 智能语音识别工具

> **声沫**，取"声音如沫，过滤提炼"之意。基于腾讯云 ASR + pyannote.audio，打造一款真正好用的语音转文字工具。

## ✨ 核心特性

| # | 特性 | 实现方案 |
|---|------|---------|
| 1 | **说话人分离** | pyannote.audio 本地分离 / 腾讯云 SpeakerDiarization 双方案 |
| 2 | **只保留我的声音** | resemblyzer 声纹注册 + 余弦相似度匹配，自动过滤非本人片段 |
| 3 | **语气词/停顿词删除** | 腾讯云 FilterModal 严格过滤 + 本地正则兜底 |
| 4 | **口误/反悔纠正** | 腾讯云口语转书面语 (ResTextFormat=5) + 本地规则 + 可选 LLM 润色 |
| 5 | **发音纠正** | 替换词表 + pypinyin 平翘舌/前后鼻音同音纠错 |
| 6 | **热词配置** | 腾讯云 HotwordList 临时热词 + 超级热词(权重11) |
| 7 | **中英文混杂** | 使用 `16k_zh_en` 中英粤混合识别引擎 |

## 📦 安装

```bash
# 基础安装（仅腾讯云 ASR + 文本后处理）
pip install -e .

# 完整安装（含说话人分离 + 声纹识别）
pip install -e ".[all]"

# 按需安装
pip install -e ".[diarization]"  # 仅说话人分离
pip install -e ".[voiceprint]"   # 仅声纹过滤
```

## ⚡ 快速开始

### 1. 配置密钥

```bash
cp .env.example .env
# 编辑 .env 填入腾讯云 SecretId/SecretKey/AppId
```

### 2. 转录音频

```bash
# 基础转录
shengmo transcribe meeting.wav

# 只保留我的声音（需先注册声纹）
shengmo transcribe meeting.wav --only-me

# 保留所有人，输出 SRT 字幕
shengmo transcribe meeting.wav --all-speakers -f srt

# 启用 LLM 润色
shengmo transcribe meeting.wav --llm-polish
```

### 3. 注册声纹

```bash
# 提供 2-3 段你说话的音频 (10-30s，清晰语音)
shengmo register my_voice1.wav my_voice2.wav
```

### 4. 查看配置

```bash
shengmo show-config
```

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────┐
│                   ShengMo Pipeline                  │
├─────────────┬──────────────┬──────────┬─────────────┤
│  ① 音频预处理  │  ② ASR 识别   │  ③ 后处理  │  ④ 输出     │
│             │              │          │             │
│ ┌─────────┐ │ ┌──────────┐ │ ┌──────┐ │ ┌─────────┐ │
│ │格式标准化│ │ │一句话识别│ │ │语气词│ │ │Markdown │ │
│ │16kHz/Mono│ │ │ (≤60s)  │ │ │过滤  │ │ │  Text   │ │
│ └─────────┘ │ ├──────────┤ │ ├──────┤ │ │  SRT    │ │
│ ┌─────────┐ │ │录音文件  │ │ │口误  │ │ │  JSON   │ │
│ │说话人   │ │ │识别(异步)│ │ │纠正  │ │ └─────────┘ │
│ │分离     │ │ ├──────────┤ │ ├──────┤ │             │
│ │pyannote │ │ │实时识别  │ │ │发音  │ │             │
│ └─────────┘ │ │WebSocket │ │ │纠正  │ │             │
│ ┌─────────┐ │ └──────────┘ │ ├──────┤ │             │
│ │声纹过滤 │ │              │ │LLM   │ │             │
│ │resemblyzer│              │ │润色  │ │             │
│ └─────────┘ │              │ └──────┘ │             │
└─────────────┴──────────────┴──────────┴─────────────┘
```

## 📋 处理流程

```
音频文件
  │
  ├─ ① 标准化: 转 16kHz / 单声道 / WAV
  │
  ├─ ② 说话人分离 (pyannote.audio)
  │    └─ 输出: [Speaker_00: 0-5s], [Speaker_01: 5-8s], ...
  │
  ├─ ③ 声纹匹配 (resemblyzer)
  │    └─ 对比每个片段与我的声纹 → 过滤非本人
  │
  ├─ ④ ASR 识别 (腾讯云)
  │    ├─ 引擎: 16k_zh_en (中英混合)
  │    ├─ 热词: DeepSeek|11, Kubernetes|10, ...
  │    ├─ 语气词过滤: FilterModal=2
  │    └─ 口语转书面语: ResTextFormat=5
  │
  ├─ ⑤ 文本后处理
  │    ├─ 发音纠正: 只持→支持, 经长→经常
  │    ├─ 语气词补充过滤: 嗯、啊、那个、就是说
  │    ├─ 口误纠正: "不是X，是Y" → Y
  │    └─ LLM 润色 (可选)
  │
  └─ ⑥ 输出: Markdown / Text / SRT / JSON
```

## 🔧 配置详解

### 热词配置

在 `config.yaml` 中配置热词：

```yaml
hotwords:
  enabled: true
  words:
    - word: "DeepSeek"
      weight: 11    # 超级热词，识别率最高
    - word: "Kubernetes"
      weight: 10    # 普通热词
```

- **权重 1-10**: 普通热词，提升识别率
- **权重 11**: 超级热词，开启同音替换，识别率极高（不宜过多，建议 < 20 个）

### 发音纠正

在 `config.yaml` 中配置常见错误：

```yaml
pronunciation_fix:
  enabled: true
  replacements:
    "只持": "支持"       # 平翘舌: zh/z 混淆
    "是线": "实现"       # 平翘舌: sh/s 混淆
    "经长": "经常"       # 前后鼻音: ng/n 混淆
```

### 声纹配置

```yaml
voiceprint:
  enabled: true
  my_voice_samples:
    - "voiceprints/my_voice_1.wav"
    - "voiceprints/my_voice_2.wav"
  similarity_threshold: 0.78   # 0.75-0.85，越高越严格
```

## 🐍 Python API

```python
from shengmo.config import load_config
from shengmo.pipeline import ShengMoPipeline

# 加载配置
config = load_config("config.yaml")

# 创建 Pipeline
pipeline = ShengMoPipeline(config)

# 注册声纹
pipeline.register_voiceprint(["my_voice1.wav", "my_voice2.wav"])

# 转录音频
result = pipeline.process(
    "meeting.wav",
    only_my_voice=True,    # 只保留我的声音
    output_path="output/meeting.md",
)

# 访问结果
print(f"时长: {result.audio_duration:.1f}s")
print(f"说话人: {result.speakers_count}")
print(f"我的片段: {result.my_segments_count}")
for r in result.results:
    print(f"[{r.speaker}] {r.start_time:.1f}s: {r.text}")
```

## 📁 项目结构

```
shengmo/
├── config.yaml          # 主配置文件
├── .env.example         # 环境变量模板
├── requirements.txt     # 依赖列表
├── pyproject.toml       # 项目元数据
├── README.md
├── voiceprints/         # 声纹样本目录
├── output/              # 输出目录
└── shengmo/             # 源码
    ├── __init__.py
    ├── config.py          # 配置管理
    ├── audio_processor.py # 音频预处理 (分离/声纹)
    ├── asr_engine.py      # 腾讯云 ASR 封装
    ├── text_processor.py  # 文本后处理
    ├── formatter.py       # 输出格式化
    ├── pipeline.py        # 核心 Pipeline
    └── cli.py             # CLI 入口
```

## 🔑 腾讯云 ASR 参数对照

| 你的需求 | 腾讯云参数 | 值 |
|---------|-----------|---|
| 中英混合识别 | `EngineModelType` | `16k_zh_en` |
| 说话人分离 | `SpeakerDiarization` | `1` |
| 语气词过滤 | `FilterModal` | `2` (严格) |
| 口语转书面语 | `ResTextFormat` | `5` (增值服务) |
| 热词增强 | `HotwordList` | `"DeepSeek\|11,K8s\|10"` |
| 词级别时间戳 | `WordInfo` | `2` |
| 脏词过滤 | `FilterDirty` | `1` |

## ⚠️ 注意事项

1. **腾讯云口语转书面语** (`ResTextFormat=5`) 是增值付费功能，需要在控制台开通
2. **pyannote.audio** 模型需要 HuggingFace Token，首次使用需下载模型 (~1GB)
3. **声纹样本**建议 2-3 段，每段 10-30 秒，安静环境录制
4. **超级热词** (权重=11) 不宜超过 20 个，否则影响整体识别率
5. 录音文件识别是异步接口，长音频处理可能需要等待几分钟

## 📄 License

MIT
