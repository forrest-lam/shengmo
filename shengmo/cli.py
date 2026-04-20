"""
声墨 (ShengMo) CLI 入口
"""

import sys
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .config import load_config
from .pipeline import ShengMoPipeline

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="shengmo")
def cli():
    """🎙️ 声墨 (ShengMo) - 智能语音识别工具"""
    pass


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-c", "--config", "config_path", default="config.yaml", help="配置文件路径")
@click.option("-o", "--output", "output_path", default=None, help="输出文件路径")
@click.option("-f", "--format", "output_format", default=None,
              type=click.Choice(["text", "srt", "json", "markdown"]),
              help="输出格式")
@click.option("--only-me/--all-speakers", default=True, help="只保留我的声音 / 保留所有人")
@click.option("--no-diarization", is_flag=True, help="禁用说话人分离")
@click.option("--no-filter", is_flag=True, help="禁用语气词过滤")
@click.option("--no-correction", is_flag=True, help="禁用口误纠正")
@click.option("--llm-polish", is_flag=True, help="启用 LLM 润色")
@click.option("-v", "--verbose", is_flag=True, help="详细日志")
def transcribe(audio_path, config_path, output_path, output_format,
               only_me, no_diarization, no_filter, no_correction, llm_polish, verbose):
    """
    转录音频文件

    示例:
        shengmo transcribe meeting.wav
        shengmo transcribe call.mp3 --only-me -f srt
        shengmo transcribe lecture.m4a --all-speakers --llm-polish
    """
    setup_logging(verbose)

    # 加载配置
    config = load_config(config_path)

    # CLI 参数覆盖
    if output_format:
        config.output.format = output_format
    if no_diarization:
        config.speaker_diarization.enabled = False
    if no_filter:
        config.filler_filter.enabled = False
    if no_correction:
        config.correction.enabled = False
    if llm_polish:
        config.correction.use_llm_polish = True

    # 运行 Pipeline
    console.print(Panel.fit(
        f"🎙️ [bold]声墨 ASR[/bold]\n"
        f"📁 音频: {audio_path}\n"
        f"👤 只保留我的声音: {'是' if only_me else '否'}\n"
        f"📝 输出格式: {config.output.format}",
        title="开始处理",
    ))

    try:
        pipeline = ShengMoPipeline(config)
        result = pipeline.process(audio_path, only_my_voice=only_me, output_path=output_path)

        # 展示结果摘要
        table = Table(title="处理结果")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        table.add_row("音频时长", f"{result.audio_duration:.1f}s")
        table.add_row("说话人数", str(result.speakers_count))
        table.add_row("我的片段", str(result.my_segments_count))
        table.add_row("识别段数", str(len(result.results)))
        table.add_row("输出文件", result.output_path)
        console.print(table)

        # 打印前几段结果预览
        console.print("\n[bold]📄 结果预览:[/bold]")
        preview = result.formatted_output[:1000]
        if len(result.formatted_output) > 1000:
            preview += "\n... (已截断)"
        console.print(preview)

    except Exception as e:
        console.print(f"[red]❌ 处理失败: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("audio_paths", nargs=-1, type=click.Path(exists=True))
@click.option("-c", "--config", "config_path", default="config.yaml", help="配置文件路径")
@click.option("-v", "--verbose", is_flag=True, help="详细日志")
def register(audio_paths, config_path, verbose):
    """
    注册我的声纹

    提供一段或多段你说话的音频，系统会提取声纹特征用于后续的声音过滤。
    建议提供 2-3 段 10-30 秒的清晰语音。

    示例:
        shengmo register my_voice1.wav my_voice2.wav
    """
    setup_logging(verbose)

    if not audio_paths:
        console.print("[red]请提供至少一个声纹音频文件[/red]")
        sys.exit(1)

    config = load_config(config_path)
    pipeline = ShengMoPipeline(config)

    console.print(f"🎤 注册声纹，使用 {len(audio_paths)} 个样本...")
    success = pipeline.register_voiceprint(list(audio_paths))

    if success:
        console.print("[green]✅ 声纹注册成功！[/green]")
        console.print("  请将以下路径添加到 config.yaml 的 voiceprint.my_voice_samples:")
        for p in audio_paths:
            console.print(f"    - {p}")
    else:
        console.print("[red]❌ 声纹注册失败，请检查音频文件[/red]")
        sys.exit(1)


@cli.command()
@click.option("-c", "--config", "config_path", default="config.yaml", help="配置文件路径")
def show_config(config_path):
    """显示当前配置"""
    config = load_config(config_path)

    table = Table(title="声墨配置")
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="green")

    table.add_row("引擎模型", config.engine.model_type)
    table.add_row("采样率", str(config.engine.sample_rate))
    table.add_row("说话人分离", f"{'启用' if config.speaker_diarization.enabled else '禁用'} "
                               f"({config.speaker_diarization.backend})")
    table.add_row("声纹过滤", f"{'启用' if config.voiceprint.enabled else '禁用'} "
                              f"(阈值: {config.voiceprint.similarity_threshold})")
    table.add_row("热词", f"{'启用' if config.hotwords.enabled else '禁用'} "
                         f"({len(config.hotwords.words)} 个词)")
    table.add_row("语气词过滤", f"{'启用' if config.filler_filter.enabled else '禁用'} "
                               f"(等级: {config.filler_filter.cloud_filter_level})")
    table.add_row("口误纠正", f"{'启用' if config.correction.enabled else '禁用'}")
    table.add_row("口语转书面语", f"{'启用' if config.correction.use_cloud_oral2written else '禁用'}")
    table.add_row("LLM 润色", f"{'启用' if config.correction.use_llm_polish else '禁用'}")
    table.add_row("发音纠正", f"{'启用' if config.pronunciation_fix.enabled else '禁用'} "
                              f"({len(config.pronunciation_fix.replacements)} 条规则)")
    table.add_row("输出格式", config.output.format)
    table.add_row("腾讯云", f"{'已配置' if config.tencent_cloud.secret_id else '未配置'}")

    console.print(table)

    if config.hotwords.words:
        hw_table = Table(title="热词列表")
        hw_table.add_column("热词", style="cyan")
        hw_table.add_column("权重", style="yellow")
        hw_table.add_column("类型", style="green")
        for w in config.hotwords.words:
            hw_type = "🔥 超级热词" if w.weight == 11 else "普通热词"
            hw_table.add_row(w.word, str(w.weight), hw_type)
        console.print(hw_table)


@cli.command()
@click.argument("words", nargs=-1)
@click.option("-w", "--weight", default=10, type=int, help="热词权重 (1-11, 11=超级热词)")
def add_hotword(words, weight):
    """
    快速添加热词

    示例:
        shengmo add-hotword DeepSeek Harness --weight 11
    """
    if not words:
        console.print("[red]请提供要添加的热词[/red]")
        return

    for w in words:
        hw_type = "超级热词 🔥" if weight == 11 else "普通热词"
        console.print(f"  + {w} (权重={weight}, {hw_type})")

    console.print("\n[yellow]请将以下内容添加到 config.yaml 的 hotwords.words:[/yellow]")
    for w in words:
        console.print(f'    - word: "{w}"')
        console.print(f'      weight: {weight}')


def main():
    cli()


if __name__ == "__main__":
    main()
