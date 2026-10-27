import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Wrap query tokens (3+ chars, diacritics-insensitive) in <mark> for a snippet. */
export function highlight(text: string, query: string): (string | { m: string })[] {
  const fold = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const toks = Array.from(new Set(fold(query).split(/[^\p{L}\p{N}']+/u).filter((t) => t.length >= 3)));
  if (!toks.length) return [text];
  const folded = fold(text);
  // the fold keeps length only if no combining marks were removed; map through NFD indices otherwise
  const nfd = text.normalize("NFD");
  const map: number[] = []; // index in nfd -> index in folded
  let j = 0;
  for (let i = 0; i < nfd.length; i++) {
    const ch = nfd[i];
    if (/[̀-ͯ]/.test(ch)) {
      map.push(j);
      continue;
    }
    map.push(j);
    j++;
  }
  const ranges: [number, number][] = [];
  for (const t of toks) {
    let from = 0;
    while (true) {
      const at = folded.indexOf(t, from);
      if (at < 0) break;
      ranges.push([at, at + t.length]);
      from = at + t.length;
    }
  }
  if (!ranges.length) return [text];
  ranges.sort((a, b) => a[0] - b[0]);
  // merge overlaps, then convert folded ranges back to nfd ranges
  const merged: [number, number][] = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([...r]);
  }
  const inv = (fi: number) => {
    let i = map.findIndex((v) => v >= fi);
    if (i < 0) i = nfd.length;
    return i;
  };
  const out: (string | { m: string })[] = [];
  let cur = 0;
  for (const [a, b] of merged) {
    const na = inv(a);
    let nb = inv(b);
    while (nb < nfd.length && /[̀-ͯ]/.test(nfd[nb])) nb++;
    if (na > cur) out.push(nfd.slice(cur, na).normalize("NFC"));
    out.push({ m: nfd.slice(na, nb).normalize("NFC") });
    cur = nb;
  }
  if (cur < nfd.length) out.push(nfd.slice(cur).normalize("NFC"));
  return out;
}
