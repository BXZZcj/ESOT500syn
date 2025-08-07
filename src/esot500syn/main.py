# src/esot500syn/main.py
import tyro
from dataclasses import dataclass
from pathlib import Path

from .recorder import Recorder

@dataclass
class Args:
    config_path: Path
    """Path to the .yaml configuration file for the recording session."""

def main():
    """主程序入口"""
    # 使用 tyro 解析命令行参数
    args = tyro.cli(Args)
    
    # 检查配置文件是否存在
    if not args.config_path.exists():
        print(f"Error: Configuration file not found at '{args.config_path}'")
        return

    # 初始化并运行 Recorder
    recorder = Recorder(config_path=str(args.config_path))
    recorder.run()

if __name__ == '__main__':
    main()