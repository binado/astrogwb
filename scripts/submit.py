import shlex
import subprocess
from typing import Sequence


class JobSubmitter:
    runner: Sequence[str]
    script: str
    modules: Sequence[str]
    env: dict[str, str]

    def __init__(self, flags: Sequence[str] | None = None):
        self._flags = flags or []

    def script_args(self) -> Sequence[str]:
        raise NotImplementedError("Subclasses must implement get_script_args()")

    def build_input(self) -> str:
        lines = []
        for module in self.modules:
            lines.append(f"module load {module}")

        script_line = shlex.join(self.runner + [self.script] + self.script_args())
        lines.append(script_line)
        return "\n".join(lines)

    @property
    def flags(self) -> Sequence[str]:
        return self._flags

    def submit(self) -> None:
        input_str = self.build_input()
        subprocess.run(
            [
                "sbatch",
                *self.flags,
            ],
            input=input_str,
            text=True,
        )


class GenerateInjectionWaveformsSubmitter(JobSubmitter):
    runner = ["python", "generate_injection_waveforms.py"]
    modules = ["miniconda/24.4.0-libmamba"]

    def __init__(
        self, input_file: str, batch_size: int = 1, flags: Sequence[str] | None = None
    ):
        self.input_file = input_file

        nlines = sum(1 for _ in open(input_file)) - 1  # exclude header
        self.njobs = (nlines + batch_size - 1) // batch_size  # ceil division
        flags += ["--array=0-{}".format(self.njobs - 1)]
        super().__init__(flags)


def submit_generate_injection_waveforms(
    input_file: str, modules: Sequence[str] | None = None, num_cpus: int = 1
):
    nlines = sum(1 for _ in open(input_file)) - 1  # exclude header
    batch_size = (nlines + num_cpus - 1) // num_cpus  # ceil division
    subprocess.run(
        [
            "sbatch",
            "--array=1-{}%{}".format(nlines, num_cpus),
            "generate_injection_waveforms.sh",
            input_file,
            str(batch_size),
        ]
    )
