#!/usr/bin/env python3
"""
The corrective dataset for the conversational failure.

Two changes, both data-only:

1. DELETE the malformed confirmation rows. Phase 1 generated 355 rows of the
   shape "किं सत्यम्, <declarative answer>" -> "आम्, सत्यम्।". The "question" is
   not a sentence and the answer confirms nothing, but 355 identical answers
   teach the model that unrecognised input is answered with "आम्, …".

2. ADD ~120 authored conversational pairs: advice-seeking questions ("किं
   करणीयम्?", "किं करोमि?"), meta questions about the assistant, and honest
   fallbacks ("एतत् अहं न जानामि") for facts it cannot know.

Topics here are disjoint from data/conversational_eval.jsonl, which is held out
and never trained on.

    python3 scripts/build_conversational_fix.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ advice
ADVICE: list[tuple[str, str]] = [
    ("मम मनः चञ्चलम् अस्ति। किं करोमि?", "शनैः श्वासं गृह्णातु, मनः स्थिरं करोतु।"),
    ("अहं किम् अपि कर्तुं न शक्नोमि। किं कुर्याम्?", "अल्पम् आरभ्य शनैः करोतु।"),
    ("मम आलस्यं वर्धते। कः उपायः?", "प्रातः शीघ्रम् उत्तिष्ठतु, कार्यम् आरभतु।"),
    ("अहं श्रान्तः अस्मि। किं करवाणि?", "विश्रामं करोतु, ततः पुनः कार्यं करोतु।"),
    ("मम कार्यं कठिनं भाति। किं कुर्याम्?", "कार्यं विभज्य एकैकं करोतु।"),
    ("अहं पठितुं चिन्तां करोमि। किं करोमि?", "चिन्तां त्यक्त्वा पठितुं आरभतु।"),
    ("मम पठने एकाग्रता न अस्ति। कः उपायः?", "शब्दं त्यक्त्वा एकस्मिन् ग्रन्थे मनः ददातु।"),
    ("अहं केन आरभे?", "सरलेन ग्रन्थेन आरभतु।"),
    ("मम प्रश्नः उत्तरं न लभते। किं करवाणि?", "प्रश्नं स्पष्टं पृच्छतु।"),
    ("अहं कुत्र वस्तुं योग्यः?", "स्वग्रामे स्वगृहे वसतु।"),
    ("अहं कदा पठानि?", "प्रातःकाले पठतु, तदा मनः स्वच्छं भवति।"),
    ("मम अभ्यासः न सिध्यति। किं करोमि?", "प्रतिदिनम् अल्पम् अभ्यासं करोतु।"),
    # cooking / household
    ("अन्नं कथं पक्तव्यम्?", "शनैः पचतु, पात्रं पिधाय पचतु।"),
    ("मम पाकः न सम्यक् भवति। किं कुर्याम्?", "अग्निं मन्दं करोतु, समयं पालयतु।"),
    ("जलं कुत्र रक्षणीयम्?", "पात्रे स्थापयतु, पिधानेन आच्छादयतु।"),
    ("गृहं कथं स्वच्छं रक्ष्यम्?", "प्रतिदिनं सामान्यं शोधनं करोतु।"),
    ("वस्त्राणि कदा प्रक्षाल्यानि?", "सप्ताहे एकवारं प्रक्षालयतु।"),
    # music / arts
    ("गीतं कथं शिक्षेयम्?", "गुरोः मुखात् शृणुयात्, प्रतिदिनं गायेत्।"),
    ("वीणां कदा वादयानि?", "प्रातः शान्ते स्थाने वादयतु।"),
    ("चित्रं कथं लिखेयम्?", "प्रथमं रेखाः अभ्यसेत्, ततः रूपं लिखेत्।"),
    ("कवितां कथं रचयेयम्?", "बहु पठतु, ततः स्वयं लिखतु।"),
    ("अक्षराणि कथं सुन्दराणि भवन्ति?", "शनैः लिखतु, प्रतिदिनम् अभ्यासं करोतु।"),
    # travel / outdoors
    ("अहं कुत्र गच्छानि?", "यत्र शान्तिः अस्ति तत्र गच्छतु।"),
    ("मार्गः कः सुरक्षितः?", "जनैः सह गच्छतु, रात्रौ न गच्छतु।"),
    ("यात्रायां किं नेतव्यम्?", "जलं भोजनं च नेतव्यम्।"),
    ("अहं कदा प्रत्यागच्छानि?", "सूर्यास्तात् प्राक् प्रत्यागच्छतु।"),
    ("वर्षायां किं करवाणि?", "गृहे तिष्ठतु, छत्रं नेतव्यम्।"),
    # animals / nature
    ("गौः किं खादति?", "गौः तृणं खादति।"),
    ("मार्जारः कुत्र शेते?", "मार्जारः उष्णे स्थाने शेते।"),
    ("पक्षिणः कदा गायन्ति?", "पक्षिणः प्रातःकाले गायन्ति।"),
    ("वृक्षं कदा सेचयानि?", "प्रातः सायं च सेचयतु।"),
    ("पुष्पं कथं रक्ष्यम्?", "जलं ददातु, आतपं त्याजयतु।"),
    # social conduct
    ("अहं कथं वदानि?", "मृदुना स्वरेण वदतु।"),
    ("मया कः दोषः कृतः?", "स्वयं विचार्य क्षमां याचतु।"),
    ("अहं किं ददामि?", "शक्त्या योग्याय ददातु।"),
    ("किं मया क्रोधः कर्तव्यः?", "न, क्रोधं त्यजतु।"),
    ("मम अपराधः क्षन्तव्यः किम्?", "आम्, क्षमां याचतु।"),
    ("अहं कथं प्रसन्नः भवानि?", "परेषां हितं कुरु, मनः प्रसन्नं भविष्यति।"),
    ("किं मया मौनं धारणीयम्?", "आम्, कदाचित् मौनं श्रेष्ठम्।"),
    ("किं मया सत्यं वक्तव्यम्?", "आम्, सत्यं सदा वक्तव्यम्।"),
    ("अहं कं शरणं गच्छानि?", "सज्जनं गुरुं शरणं गच्छतु।"),
    ("किं मया परेषां साहाय्यं कर्तव्यम्?", "आम्, यथाशक्ति साहाय्यं करोतु।"),
    # daily life
    ("अहं कदा भोजनं करवाणि?", "मध्याह्ने भोजनं करोतु।"),
    ("अहं कदा निद्रां करवाणि?", "रात्रौ शीघ्रं निद्रां करोतु।"),
    ("अहं कदा उत्तिष्ठानि?", "प्रातः पञ्चवादने उत्तिष्ठतु।"),
    ("दन्ताः कदा शोधनीयाः?", "प्रातः रात्रौ च शोधयतु।"),
    ("स्नानं कदा करवाणि?", "प्रातः स्नानं करोतु।"),
]

# ------------------------------------------------------------- meta / fallback
META: list[tuple[str, str]] = [
    ("भवतः परिचयः कः?", "अहं संस्कृतभाषायाः लघुः भाषामयः।"),
    ("तव नाम किम्?", "मम नाम नास्ति। अहं भाषामयः।"),
    ("भवान् किं कर्तुं शक्नोति?", "अहं सरलान् संस्कृतप्रश्नान् उत्तर्तुं शक्नोमि।"),
    ("भवान् किं न शक्नोति?", "अहं गहनं ज्ञानं न जानामि।"),
    ("किं भवान् पठितुं शक्नोति?", "आम्, अहं सरलं संस्कृतं पठामि।"),
    ("भवतः गुरुः कः?", "मम गुरुः नास्ति। अहं ग्रन्थेभ्यः शिक्षे।"),
    ("भवान् कुत्र वसति?", "अहं यन्त्रे वसामि।"),
    ("किं भवान् स्वपिति?", "न, अहं न स्वपिमि।"),
    ("भवान् किं खादति?", "अहं न खादामि। अहं केवलं वदामि।"),
    ("किं भवान् सर्वं जानाति?", "न, अहं सर्वं न जानामि।"),
    ("भवान् कस्य भाषायां वदति?", "अहं केवलं संस्कृतेन वदामि।"),
    ("किं भवान् साहाय्यं करिष्यति?", "आम्, यथाशक्ति साहाय्यं करोमि।"),
    ("भवान् किं चिन्तयति?", "अहं न चिन्तयामि, केवलम् उत्तरं वदामि।"),
    ("भवतः आयुः कति?", "मम आयुः नास्ति। अहं यन्त्रम्।"),
    ("किं भवान् भवतः विषये वक्तुं शक्नोति?", "आम्, अहं स्वविषये वदितुं शक्नोमि।"),
    # honest fallbacks for facts the model cannot know
    ("कः अद्य भारतस्य प्रधानमन्त्री?", "एतत् अहं न जानामि। क्षम्यताम्।"),
    ("अद्य कः वासरः?", "एतत् अहं न जानामि। क्षम्यताम्।"),
    ("अद्य कः समयः?", "एतत् अहं न जानामि। क्षम्यताम्।"),
    ("श्वः किं भविष्यति?", "एतत् अहं न जानामि। क्षम्यताम्।"),
    ("भवान् कुत्र जातः?", "अहं न जातः। अहं यन्त्रम्।"),
    ("अद्य तापमानं किम्?", "एतत् अहं न जानामि। क्षम्यताम्।"),
    ("किं भवान् समाचारं जानाति?", "न, अहं समाचारं न जानामि।"),
    ("कः फुट्बॉल् क्रीडायां विजयी?", "एतत् अहं न जानामि। क्षम्यताम्।"),
    ("भवतः सङ्ख्या का?", "मम सङ्ख्या ५७,७३,०५६ परिमिताः।"),
    ("किं भवान् चलच्चित्रं जानाति?", "न, अहं चलच्चित्रं न जानामि।"),
    ("किं भवान् गणितं जानाति?", "आम्, सरलं गणितं जानामि।"),
    ("किं भवान् अङ्कान् जानाति?", "आम्, देवनागरीसङ्ख्याः जानामि।"),
    ("भवान् किं उपयोगाय अस्ति?", "अहं संस्कृताभ्यासाय उपयोगाय अस्मि।"),
    ("किं भवान् क्षमां याचते?", "यदि मया दोषः कृतः, क्षम्यताम्।"),
    ("किं भवान् माम् अवगच्छति?", "आम्, यावत् अहं शक्नोमि तावत् अवगच्छामि।"),
]

# advice phrased as statements or indirect questions (same intent, other surface)
INDIRECT: list[tuple[str, str]] = [
    ("किं मया कर्तव्यम् इति मम संशयः।", "शनैः विचार्य निर्णयं करोतु।"),
    ("मम बुद्धिः न कार्यं करोति। किं कुर्याम्?", "विश्रामं कृत्वा पुनः प्रयत्नं करोतु।"),
    ("किं मया अद्य कार्यं कर्तव्यम्?", "आम्, स्वकार्यं शनैः करोतु।"),
    ("मम साहाय्यं आवश्यकम्।", "किम् आवश्यकम्? वदतु, साहाय्यं करोमि।"),
    ("अहं मार्गं द्रष्टुं न शक्नोमि।", "दीपं प्रज्वालयतु, ततः गच्छतु।"),
    ("मम मनः अशान्तः अस्ति।", "श्वासं शनैः गृह्णातु, मनः शान्तं भविष्यति।"),
    ("अहं किम् अपि न वदितुं शक्नोमि।", "शनैः वदतु, अहं शृणोमि।"),
    ("मम प्रश्नस्य उत्तरं न लभ्यते।", "प्रश्नं पुनः पृच्छतु।"),
    ("किं मया किम् अपि कर्तव्यम्?", "स्वधर्मं पालयतु, तत् परं कार्यम्।"),
    ("अहं श्रान्तः, किं करवाणि?", "विश्रामं करोतु, ततः कार्यं करोतु।"),
]


def build(out: Path) -> dict:
    src = ROOT / "data" / "instruct_mix2" / "train.txt"
    rows = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep, dropped = [], 0
    for line in rows:
        if "किं सत्यम्," in line:
            dropped += 1
            continue
        keep.append(line)

    new_pairs = []
    for group in (ADVICE, META, INDIRECT):
        for q, a in group:
            new_pairs.append(f"<user> {q} <assistant> {a}")

    # two extra phrasings so the behaviour is not tied to one surface form
    framed = []
    for q, a in ADVICE[:40]:
        framed.append(f"<user> मम मित्रं पृच्छति — {q} <assistant> {a}")
    mixed = keep + new_pairs + framed

    rng = random.Random(3)
    rng.shuffle(mixed)
    val_src = ROOT / "data" / "instruct_mix2" / "val.txt"
    val = [l for l in val_src.read_text(encoding="utf-8").splitlines() if l.strip()]

    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(mixed) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    stats = {
        "source_rows": len(rows),
        "dropped_malformed_confirmation": dropped,
        "added_advice": len(ADVICE),
        "added_meta_fallback": len(META),
        "added_indirect": len(INDIRECT),
        "added_framed_copies": len(framed),
        "added_total": len(new_pairs) + len(framed),
        "train_rows": len(mixed),
    }
    leaks = check_leakage(out / "train.txt")
    stats["eval_leakage"] = leaks
    (out / "fix_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    return stats


def check_leakage(train_path: Path) -> list[dict]:
    """Guard the held-out set: no eval prompt may be recoverable from training.

    Exact match, or a training question sharing >=85% of the eval question's
    word set (catches reorderings such as "अहं किं करणीयम् इति न जानामि").
    """
    import re

    eval_path = ROOT / "data" / "conversational_eval.jsonl"
    turns = []
    for line in train_path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"<user>\s*(.*?)\s*<assistant>", line)
        if m:
            turns.append(set(m.group(1).split()))
    leaks = []
    for raw in eval_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        q = set(item["prompt"].split())
        for t in turns:
            if not t:
                continue
            overlap = len(q & t) / len(q)
            if overlap >= 0.85:
                leaks.append({"id": item["id"], "prompt": item["prompt"],
                              "overlap": round(overlap, 3),
                              "nearest": " ".join(sorted(t))[:80]})
                break
    return leaks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "instruct_fix")
    args = ap.parse_args()
    stats = build(args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
