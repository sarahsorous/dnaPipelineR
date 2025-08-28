import pandas as pd

def clean_text(text):
    return (text
            .replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-"))

def is_speaker_line(line, next_line):
    line = line.strip()
    next_line = next_line.strip()
    if not line or not next_line:
        return False
    return not line[-1:] in {".", ",", ";", ":", "!", "?"}

def extract_speaker_blocks_from_path(file_path):
    """Process a text file and return a list of dicts with speaker and speech."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    speaker_blocks = []
    current_speaker = None
    current_text = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if is_speaker_line(line, next_line):
            if current_speaker and current_text:
                speaker_blocks.append({
                    "speaker_org": current_speaker,
                    "speech": " ".join(current_text).strip()
                })
            current_speaker = line
            current_text = []
            i += 1
            continue

        if current_speaker:
            current_text.append(line)

        i += 1

    if current_speaker and current_text:
        speaker_blocks.append({
            "speaker_org": current_speaker,
            "speech": " ".join(current_text).strip()
        })

    for block in speaker_blocks:
        block["speaker_org"] = clean_text(block["speaker_org"])
        block["speech"] = clean_text(block["speech"])

    return pd.DataFrame(speaker_blocks)
