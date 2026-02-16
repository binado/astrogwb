import subprocess
from pathlib import Path


def get_config_filepath(filepath: Path, suffix: str = ".toml") -> Path:
    parent = filepath.parent
    if parent != Path(__file__).parent:
        raise ValueError(f"Config file must be in the same directory as {__file__}")
    filename = filepath.stem
    config_filepath = parent.parent / f"config/{filename}.toml"
    return config_filepath.with_suffix(suffix)


def get_git_revision() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
