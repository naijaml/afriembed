"""Stress-test the demo index with queries that have a known target article.

Each case is (label, query-or-wav, target-slug-substring). Prints the rank of the target (1 = top hit)
for the embedding model and, for typed queries, for a BM25 keyword baseline over the same chunks.
ZERO_OVERLAP holds queries that share no word with their target article: the cases that separate
semantic search from string matching. Usage:

    uv run python scripts/demo_stress_test.py [--index indexes/demo] [--adapter artifacts/audio-lora]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "demo" / "queries"

# fmt: off
TYPED = [
    # A. paraphrase within language (no headline words)
    ("hau paraphrase", "wane kamfanoni ne suka samu riba sosai daga yakin Afghanistan?", "afghanistan-manyan-kamfanonin"),
    ("hau paraphrase", "mutumin da ya fi kowa kudi a duniya yana son sayen shafin sada zumunta", "elon-musk"),
    ("hau paraphrase", "wane magani ake ba masu cutar korona?", "magungunan-hiv"),
    ("hau paraphrase", "me ya sa mutum ke jin an danne shi a cikin barci?", "dannau"),
    ("swa paraphrase", "kwa nini Wakenya hawapendi deni jipya kutoka IMF?", "mkopo-wa-imf"),
    ("pcm paraphrase", "children dey die for one Abuja village because hospital no dey", "abuja-community-wia-pikin"),
    # B. cross-lingual: English query, African-language article
    ("eng->hau", "anger in Ghana over the new electronic transaction levy", "harajin-latironi-a-ghana"),
    ("eng->yor", "parents of the conjoined twins born in Ilorin", "ilorin-conjoined-twins"),
    ("eng->ibo", "woman who put Gorilla Glue in her hair went to hospital", "gorilla-glue"),
    ("eng->swa", "Instagram will hide the number of likes on posts", "instagram-kuficha"),
    ("eng->pcm", "landlady removed the roof and door of a tenant who owed rent", "landlady-remove-roof"),
    # B'. African-language query, English or other-language article
    ("hau->eng", "ma'aikatan motar daukar marasa lafiya sun fara yajin aiki saboda albashi", "ambulance-service"),
    ("yor->hau", "Elon Musk fẹ́ ra Twitter pátápátá", "elon-musk"),
    ("ibo->pcm", "Ghana enwetala onye mbụ gbakere na Covid-19", "ghana-record-en-first-covid-19-recovery"),
    # C. diacritics stripped
    ("yor no-diacritics", "Buhari ati Osinbajo gba abere ajesara covid", "buhari-osinbajo-ti-gba"),
    ("ibo no-diacritics", "ajuju i kwesiri iju onye i choro ilu tupu unu alua", "fertility-checks"),
    # D. code-switching
    ("code-switch", "which Hausa article talks about Neymar and Liverpool wanting Antony?", "neymar"),
    # E. near-duplicate discrimination (two Hausa transfer-market round-ups)
    ("near-dup", "makomar Pogba, Bale da De Jong a kasuwar 'yan kwallo", "pogba-bale"),
    ("near-dup", "makomar Lacazette, Tielemans da Asensio", "lacazette"),
    # F. entity only
    ("entity", "Hushpuppi", "hushpuppi"),
    ("entity", "Magufuli", "magufuli"),
    ("entity", "Tessica Brown", "gorilla-glue"),
    # G. off-topic (no target; watch the top score)
    ("off-topic", "how do I replace the spark plugs in a Toyota Corolla", None),
]
ZERO_OVERLAP = [
    ("0-overlap eng->swa", "social app will stop showing how popular each photo is", "instagram-kuficha"),
    ("0-overlap eng->hau", "why people feel pinned down and unable to move while waking up", "dannau"),
    ("0-overlap eng->hau", "US defence firms that got rich from the war", "afghanistan-manyan-kamfanonin"),
    ("0-overlap eng->swa", "citizens upset that their government borrowed from an international lender", "mkopo-wa-imf"),
    ("0-overlap eng->swa", "a year since the Tanzanian president died, how his allies fared", "magufuli"),
    ("0-overlap eng->pcm", "owner of a house stripped the ceiling off a renter who had not paid", "landlady-remove-roof"),
    ("0-overlap hau->yor", "iyayen 'yan biyu da aka haifa a hade a jihar Kwara", "ilorin-conjoined-twins"),
    ("0-overlap hau->pcm", "yara na mutuwa a wani kauye kusa da babban birnin kasa saboda babu asibiti", "abuja-community-wia-pikin"),
    ("0-overlap swa->ibo", "mwanamke aliyepaka gundi kali kwenye nywele akaenda hospitalini", "gorilla-glue"),
    ("0-overlap yor->hau", "ọlọ́rọ̀ jùlọ lágbàáyé fẹ́ ra ilé iṣẹ́ ayélujára kan", "elon-musk-mai"),
    ("0-overlap eng->yor", "president and his deputy got their covid jab", "buhari-osinbajo-ti-gba"),
    ("0-overlap eng->ibo", "questions to ask your partner before marriage about having children", "fertility-checks"),
]
SPOKEN = [
    ("spoken hau", "hau_afghanistan_companies.wav", "afghanistan-manyan-kamfanonin"),
    ("spoken hau", "hau_richest_man_twitter.wav", "elon-musk"),
    ("spoken hau->eng", "hau_ambulance_strike.wav", "ambulance-service"),
    ("spoken yor", "yor_buhari_vaccine.wav", "buhari-osinbajo-ti-gba"),
    ("spoken ibo", "ibo_covid_recovery.wav", "coronavirus-recovery"),
    ("spoken swh", "swh_kenya_imf_loan.wav", "mkopo-wa-imf"),
    ("spoken hau (FLEURS, real)", "fleurs_hau_105_mosquito_virus.wav", None),
]
# fmt: on


class BM25:
    """Keyword baseline over the same chunks (diacritics folded), to show what string matching gets."""

    def __init__(self, meta: list[dict]):
        import math
        import re
        import unicodedata
        from collections import Counter

        self.meta = meta

        def fold(t):
            return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if not unicodedata.combining(c))

        self.toks = lambda t: re.findall(r"[\w']+", fold(t))
        self.docs = [Counter(self.toks(m.get("text") or "")) for m in meta]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def __call__(self, q: str, k: int = 40) -> list[dict]:
        qt = self.toks(q)
        sc = []
        for d, L in zip(self.docs, self.len, strict=False):
            s = 0.0
            for t in qt:
                if t in d:
                    s += self.idf[t] * d[t] * 2.5 / (d[t] + 1.5 * (0.25 + 0.75 * L / self.avg))
            sc.append(s)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [{"score": sc[i], **self.meta[i]} for i in order]


def rank(hits: list[dict], target: str | None, k: int = 10) -> str:
    """Rank of the target among distinct documents (chunks of one article collapse to one row)."""
    if target is None:
        return "-"
    seen: list[str] = []
    for h in hits:
        f = h.get("file", "")
        if f in seen:
            continue
        seen.append(f)
        if target in (h.get("title") or f):
            return str(len(seen))
        if len(seen) >= k:
            break
    return f">{k}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="indexes/demo")
    ap.add_argument("--adapter", default="artifacts/audio-lora")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    from afriembed.experiments.common import ModelOpts
    from afriembed.experiments.search import Searcher

    s = Searcher(ModelOpts(model="embeddinggemma-2", adapter=a.adapter), Path(a.index), need_audio=True)
    bm = BM25(s.idx.meta)

    def block(title, cases):
        print(f"\n== {title}\n{'embed':>6} {'bm25':>5}  {'case':<22} query")
        e = b = n = 0
        for label, q, target in cases:
            hits = s(query=q, k=a.k * 4)
            re_, rb = rank(hits, target, a.k), rank(bm(q), target, a.k)
            top = " | ".join(f"{h['score']:.2f} {(h.get('title') or '')[:30]}" for h in hits[:2])
            print(f"{re_:>6} {rb:>5}  {label:<22} {q[:58]}\n{'':13}{top}")
            if target:
                n += 1
                e += re_ == "1"
                b += rb == "1"
        print(f"   target at rank 1: embeddings {e}/{n}, bm25 {b}/{n}")

    block("typed, target shares some words with the query", TYPED)
    block("typed, zero word overlap with the target (semantic / cross-lingual)", ZERO_OVERLAP)

    print(f"\n== spoken (no text at all; keyword search cannot run these)\n{'embed':>6}  {'case':<26} file")
    e = n = 0
    for label, wav, target in SPOKEN:
        p = Q / wav
        if not p.exists():
            print(f"{'miss':>6}  {label:<26} {wav} (file missing)")
            continue
        hits = s(audio_query=p, k=a.k * 4)
        r = rank(hits, target, a.k)
        top = " | ".join(f"{h['score']:.2f} {(h.get('title') or '')[:30]}" for h in hits[:2])
        print(f"{r:>6}  {label:<26} {wav}\n{'':8}{top}")
        if target:
            n += 1
            e += r == "1"
    print(f"   target at rank 1: {e}/{n}")


if __name__ == "__main__":
    main()
