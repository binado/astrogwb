import subprocess
from typing import Sequence


def submit_job_array(
    script: str, input_file: str, modules: Sequence[str] | None = None
):
    script_lines = []
    nlines = sum(1 for _ in open(input_file))
    if modules:
        for module in modules:
            script_lines.append("module load {}".format(module))

    subprocess.run(["sbatch", "--array=1-{}".format(nlines), script, input_file])
