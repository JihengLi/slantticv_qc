from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Union

VALID_ENTS = {"sub", "ses", "task", "acq", "ce", "rec", "run", "echo", "part", "chunk"}
ORDERED_ENTS = ["task", "acq", "ce", "rec", "run", "echo", "part", "chunk"]
ENT_PAIR_RE = re.compile(r"([a-z]+-[^_]+)")
SUFFIX_RE = re.compile(r"_[^_]+$")
NII_GZ_RE = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)


def _ensure_str(x) -> str:
    if x is None:
        raise TypeError("subject_id/stem is None")
    if isinstance(x, (str, bytes)):
        return x.decode() if isinstance(x, bytes) else x
    return str(x)


def _basename_no_ext(p: str) -> str:
    name = Path(p).name
    return NII_GZ_RE.sub("", name)


def parse_entities(stem: Union[str, Path]) -> Dict[str, str]:
    s = _ensure_str(stem)
    s = _basename_no_ext(s)
    return {
        k: v
        for k, v in (pair.split("-", 1) for pair in ENT_PAIR_RE.findall(s))
        if k in VALID_ENTS
    }


def _strip_suffix(subject_id: str) -> str:
    return SUFFIX_RE.sub("", subject_id)


def _build_slant_names(prefix: str, ents: Dict[str, str]) -> List[str]:
    bits = [f"{k}-{ents[k]}" for k in ORDERED_ENTS if k in ents]
    names = [prefix]
    if bits:
        names.append(prefix + bits[0] + "".join(f"_{b}" for b in bits[1:]))
        names.append(prefix + "_" + "_".join(bits))
    return names


def find_t1w_addr_subjectid(
    subject_id: Union[str, Path], t1_root: Union[str, Path]
) -> List[str]:
    t1_root = Path(t1_root)
    ents = parse_entities(subject_id)

    if "sub" not in ents:
        raise ValueError(f"No 'sub-' entity found in subject_id: {subject_id}")

    sub_dir = t1_root / f"sub-{ents['sub']}"
    ses_dir = sub_dir / f"ses-{ents['ses']}" if "ses" in ents else sub_dir
    anat_dir = ses_dir / "anat"

    if not anat_dir.exists():
        return []

    base = _strip_suffix(_ensure_str(subject_id))
    candidate_patterns = [f"{base}_T1w.nii.gz", "*_T1w.nii.gz"]

    results: List[str] = []
    for pat in candidate_patterns:
        results.extend(str(p) for p in anat_dir.glob(pat))

    return sorted(set(results))


def find_slant_addr_subjectid(
    subject_id: Union[str, Path], slant_root: Union[str, Path]
) -> List[str]:
    root = Path(slant_root)
    ents = parse_entities(subject_id)

    if "sub" not in ents:
        raise ValueError(f"No 'sub-' entity found in subject_id: {subject_id}")

    sub_dir = root / f"sub-{ents['sub']}"
    level_dir = sub_dir / f"ses-{ents['ses']}" if "ses" in ents else sub_dir
    if not level_dir.exists():
        return []

    slant_prefix = "SLANT-TICVv1.2"
    candidate_names = _build_slant_names(slant_prefix, ents)

    slant_dirs = [level_dir / n for n in candidate_names if (level_dir / n).is_dir()]
    if not slant_dirs:
        slant_dirs = list(level_dir.glob(f"{slant_prefix}*"))

    base = _strip_suffix(_ensure_str(subject_id))
    target_name = f"{base}_T1w_seg.nii.gz"

    hits: List[str] = []
    for sd in slant_dirs:
        fr = sd / "post" / "FinalResult"
        if not fr.is_dir():
            continue

        exact = list(fr.glob(target_name))
        if exact:
            hits.extend(str(p) for p in exact)
        else:
            hits.extend(str(p) for p in fr.glob("*_T1w_seg.nii.gz"))

    return sorted(set(hits))
