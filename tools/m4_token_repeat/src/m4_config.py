"""Frozen M4 rules and public result values shared by the native adapter."""

from dataclasses import dataclass

CONFIG_M4 = """{"components":[
{"kind":"V","mn":32,"mx":500,"c":5},
{"kind":"V","mn":32,"mx":2000,"c":3},
{"kind":"RUN","min_chars":500},
{"kind":"M","feat":"A3_n4_W1024","theta":0.6,"S":8,"nov":{"feat":"N_n4_W512","max":0.05}},
{"kind":"M","feat":"A5_n4_W512","theta":0.7,"S":4,"nov":{"feat":"N_n4_W512","max":0.1}},
{"kind":"M","feat":"A5_n4_W512","theta":0.5,"S":4,"guard":0.5,"nov":{"feat":"N_n4_W512","max":0.05}},
{"kind":"M","feat":"P_n8_W1024","theta":0.6,"S":24,"nov":{"feat":"N_n4_W512","max":0.02}},
{"kind":"M","feat":"wrep_W2048","theta":0.95,"S":16,"and":[{"feat":"mark_W1024","min":0.05}]},
{"kind":"M","feat":"wrep_W2048","theta":0.99,"S":24,"and":[]},
{"kind":"M","feat":"wrep_W1024","theta":0.99,"S":8,"and":[{"feat":"mark_W1024","min":0.04}]},
{"kind":"M","feat":"wrep_W1024","theta":0.98,"S":8,"and":[{"feat":"mark_W1024","min":0.05}]},
{"kind":"M","feat":"sdup_N16","theta":0.7,"S":24,"and":[{"feat":"wrep_W2048","min":0.9},{"feat":"mark_W1024","min":0.03}]},
{"kind":"M","feat":"sdup_N16","theta":0.6,"S":40,"and":[{"feat":"mark_W1024","min":0.03}]},
{"kind":"M","feat":"wrep_W2048","theta":0.95,"S":48,"and":[{"feat":"mark_W1024","min":0.03}]}
]}"""


@dataclass
class Result:
    is_ill: bool = False
    fired_at: int = -1  # raw-token boundary at which the alert is raised
    reason: str = ""


def describe(comp):
    k = comp["kind"]
    if k == "M":
        return (
            f"{comp['feat']}>={comp['theta']}"
            + (f" S{comp['S']}" if comp.get("S", 1) > 1 else "")
            + (f" sep<={comp['guard']}" if comp.get("guard") is not None else "")
            + (f" {comp['nov']['feat']}<={comp['nov']['max']}" if comp.get("nov") else "")
            + "".join(
                f" & {a['feat']}{'>=' if 'min' in a else '<='}{a.get('min', a.get('max'))}" for a in comp.get("and", [])
            )
        )
    if k == "V":
        return f"vllm[{comp['mn']},{comp['mx']}]x{comp['c']}"
    if k == "RUN":
        return f"run>={comp['min_chars']}"
    raise ValueError(f"Unsupported M4 component: {k}")
