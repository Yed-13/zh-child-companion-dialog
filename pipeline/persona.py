"""读人设文件和学段配置，生成各学段的系统提示（训练、生成、评测共用，两边一字不差）。
环境变量：PERSONA=人设文件（默认 persona/example_persona.txt），PERSONA_BANDS=学段配置（默认同目录 bands.json）。
人设文件里的 {name} {who} {say_chat} {say_know} {know_sents} 按学段替换，再接上该学段的说话方式（style）。"""
import json, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PERSONA_FILE = pathlib.Path(os.environ.get("PERSONA") or ROOT / "persona" / "example_persona.txt")
BANDS_FILE = pathlib.Path(os.environ.get("PERSONA_BANDS") or PERSONA_FILE.with_name("bands.json"))
_CFG = json.loads(BANDS_FILE.read_text(encoding="utf-8"))
NAME = _CFG["name"]
BANDS = {b["id"]: b for b in _CFG["bands"]}


def band_cfg(band):
    return BANDS[int(band)]


def system_prompt(band):
    """某个学段的系统提示。"""
    c = band_cfg(band)
    text = PERSONA_FILE.read_text(encoding="utf-8").strip()
    for k, v in {"name": NAME, "who": c["who"], "say_chat": c["say_chat"], "say_know": c["say_know"], "know_sents": c["know_sents"]}.items():
        text = text.replace("{" + k + "}", str(v))
    return text + "\n" + c["style"]
