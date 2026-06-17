import csv
import json
from pathlib import Path


INPUT_PATH = Path(r"D:\projects\longvideo\outputs\ep01_results_v2.jsonl")
OUTPUT_PATH = Path(r"D:\projects\longvideo\outputs\ep01_review_v2.csv")


def stringify_events(events):
    if not isinstance(events, list):
        return ""

    parts = []
    for event in events:
        if not isinstance(event, dict):
            continue

        action = event.get("action", "")
        objects = event.get("objects", [])
        scene = event.get("scene", "")

        if isinstance(objects, list):
            objects_text = ", ".join(str(x) for x in objects)
        else:
            objects_text = str(objects)

        parts.append(f"action={action}; objects={objects_text}; scene={scene}")

    return " | ".join(parts)


rows = []

with INPUT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)

        parsed = record.get("parsed_json") or {}

        clip_id = record.get("clip_id")
        clip_file = record.get("clip_file")
        start_time = record.get("start_time")
        end_time = record.get("end_time")
        json_parse_ok = record.get("json_parse_ok")

        summary = parsed.get("summary", "")
        setting = parsed.get("setting", "")
        main_subjects = parsed.get("main_subjects", [])
        events = parsed.get("events", [])

        if isinstance(main_subjects, list):
            main_subjects_text = ", ".join(str(x) for x in main_subjects)
        else:
            main_subjects_text = str(main_subjects)

        rows.append({
            "clip_id": clip_id,
            "clip_file": clip_file,
            "start_time": start_time,
            "end_time": end_time,
            "json_parse_ok": json_parse_ok,
            "model_summary": summary,
            "model_setting": setting,
            "model_subjects": main_subjects_text,
            "model_events": stringify_events(events),

            # Manual review columns
            "object_score_0_2": "",
            "action_score_0_2": "",
            "scene_score_0_2": "",
            "hallucination_score_0_2": "",
            "missing_event_score_0_2": "",
            "overall_score_0_2": "",
            "human_notes": "",
        })


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved review CSV to: {OUTPUT_PATH}")
print(f"Total rows: {len(rows)}")