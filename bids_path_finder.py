"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

import re

from pathlib import Path
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


def _build_slant_names(prefix: str, ents: Dict[str, str]) -> List[str]:
    bits = [f"{k}-{ents[k]}" for k in ORDERED_ENTS if k in ents]
    names = [prefix]
    if bits:
        names.append(prefix + bits[0] + "".join(f"_{b}" for b in bits[1:]))
        names.append(prefix + "_" + "_".join(bits))
    return names


def find_t1w_addr_subjectid(
    subject_id: Union[str, Path], t1_root: Union[str, Path]
) -> Path:
    t1_root = Path(t1_root)
    ents = parse_entities(subject_id)
    if "sub" not in ents:
        raise ValueError(f"No 'sub-' entity found in subject_id: {subject_id}")

    sub_dir = t1_root / f"sub-{ents['sub']}"
    ses_dir = sub_dir / f"ses-{ents['ses']}" if "ses" in ents else sub_dir
    anat_dir = ses_dir / "anat"
    if not anat_dir.exists():
        raise FileNotFoundError(f"anat dir not found: {anat_dir}")

    target_name = f"{_ensure_str(subject_id)}_T1w.nii.gz"
    target_path = anat_dir / target_name

    if target_path.is_file():
        return target_path

    hits = list(anat_dir.glob("*_T1w.nii.gz"))
    if not hits:
        raise FileNotFoundError(
            f"{target_name} not found for {subject_id} in {anat_dir}"
        )
    if len(hits) > 1:
        raise RuntimeError(f"Multiple T1w files for {subject_id}: {hits}")

    return hits[0]


def find_slant_addr_subjectid(
    subject_id: Union[str, Path], slant_root: Union[str, Path]
) -> Path:
    root = Path(slant_root)
    ents = parse_entities(subject_id)
    if "sub" not in ents:
        raise ValueError(f"No 'sub-' entity found in subject_id: {subject_id}")

    sub_dir = root / f"sub-{ents['sub']}"
    level_dir = sub_dir / f"ses-{ents['ses']}" if "ses" in ents else sub_dir
    if not level_dir.exists():
        raise FileNotFoundError(f"Level dir not found: {level_dir}")

    slant_prefix = "SLANT-TICVv1.2"
    cand_names = _build_slant_names(slant_prefix, ents)
    slant_dirs = [level_dir / n for n in cand_names if (level_dir / n).is_dir()]
    if not slant_dirs:
        slant_dirs = list(level_dir.glob(f"{slant_prefix}*"))
    if not slant_dirs:
        raise FileNotFoundError(f"No SLANT dir under {level_dir}")

    target_name = f"{_ensure_str(subject_id)}_T1w_seg.nii.gz"

    hits: list[Path] = []
    for sd in slant_dirs:
        fr = sd / "post" / "FinalResult"
        if fr.is_dir():
            p = fr / target_name
            if p.is_file():
                hits.append(p)

    if not hits:
        raise FileNotFoundError(f"{target_name} not found for {subject_id}")
    if len(hits) > 1:
        raise RuntimeError(f"Multiple matches for {subject_id}: {hits}")

    return hits[0]
