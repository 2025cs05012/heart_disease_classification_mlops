"""Extract mermaid code blocks from architecture.md into .mmd files for mmdc."""
import re
import pathlib

ROOT = pathlib.Path(__file__).parent
src = (ROOT / "architecture.md").read_text()
blocks = re.findall(r"```mermaid\n(.*?)```", src, flags=re.DOTALL)
print(f"found {len(blocks)} mermaid blocks")
if blocks:
    (ROOT / "_arch_flow.mmd").write_text(blocks[0])
if len(blocks) >= 2:
    (ROOT / "_arch_seq.mmd").write_text(blocks[1])
