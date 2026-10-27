"""Image-caption data in African languages.

- Afri-Aya (CohereLabsCommunity/afri-aya): images inline, caption_en + caption_local, many languages/communities.
- AfriCaption (AfriMM/AFRICaption): Flickr8k captions translated into 20 African languages; images from jxie/flickr8k.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import hf_token

log = logging.getLogger(__name__)


@dataclass
class CaptionItem:
    key: str
    image: object  # PIL.Image
    language: str
    caption_local: str
    caption_en: str


def load_afri_aya(
    max_items: int | None = None, languages: list[str] | None = None, seed: int = 0
) -> list[CaptionItem]:
    from datasets import load_dataset

    ds = load_dataset("CohereLabsCommunity/afri-aya", split="train", token=hf_token())
    if languages:
        wanted = {l.lower() for l in languages}
        ds = ds.filter(lambda r: str(r["language"]).lower() in wanted)
    if max_items and len(ds) > max_items:
        ds = ds.shuffle(seed=seed).select(range(max_items))
    out = []
    for r in ds:
        if not r.get("caption_local") or not r.get("caption_en"):
            continue
        out.append(
            CaptionItem(str(r["id"]), r["image"], str(r["language"]), r["caption_local"], r["caption_en"])
        )
    log.info("Afri-Aya: %d items, %d languages", len(out), len({o.language for o in out}))
    return out


def load_africaption(
    lang_cols: list[str], max_items: int | None = None, seed: int = 0
) -> dict[str, list[CaptionItem]]:
    """Returns {lang_col: [CaptionItem...]} with Flickr8k images joined by filename."""
    from datasets import load_dataset

    caps = load_dataset("AfriMM/AFRICaption", split="train", token=hf_token())
    if max_items and len(caps) > max_items:
        caps = caps.shuffle(seed=seed).select(range(max_items))
    wanted = {r["image_id"] for r in caps}
    images = {}
    flickr = load_dataset("jxie/flickr8k", split="train", token=hf_token())
    name_col = next(
        (c for c in flickr.column_names if c in ("image_id", "filename", "file_name", "id")), None
    )
    for r in flickr:
        name = str(r[name_col]) if name_col else None
        if name and not name.endswith(".jpg"):
            name = name + ".jpg"
        if name in wanted:
            images[name] = r["image"]
    if not images:
        raise RuntimeError(
            "Could not join AfriCaption image_ids to jxie/flickr8k; check column names: "
            + ", ".join(flickr.column_names)
        )
    out: dict[str, list[CaptionItem]] = {c: [] for c in lang_cols}
    for r in caps:
        im = images.get(r["image_id"])
        if im is None:
            continue
        for c in lang_cols:
            if r.get(c):
                out[c].append(CaptionItem(r["image_id"], im, c, r[c], r["eng"]))
    log.info("AfriCaption: %s", {c: len(v) for c, v in out.items()})
    return out
