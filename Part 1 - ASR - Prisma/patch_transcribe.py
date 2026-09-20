import sys

with open('evaluate_prisma.py', 'r') as f:
    content = f.read()

old_def = "def transcribe(audio_bytes: bytes, lang_code: str) -> tuple[str, float]:"
new_def = "def _transcribe_chunk(audio_bytes: bytes, lang_code: str) -> tuple[str, float]:"

if old_def not in content:
    print("could not find old def")
    sys.exit(1)

content = content.replace(old_def, new_def)

new_transcribe = """
def transcribe(audio_bytes: bytes, lang_code: str) -> tuple[str, float]:
    import soundfile as sf
    import io
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        total_sec = len(data) / sr
    except Exception:
        return _transcribe_chunk(audio_bytes, lang_code)
        
    MAX_DURATION = 29.0
    if total_sec <= MAX_DURATION:
        return _transcribe_chunk(audio_bytes, lang_code)
        
    transcripts = []
    total_latency = 0.0
    chunk_samples = int(MAX_DURATION * sr)
    
    for i in range(0, len(data), chunk_samples):
        chunk_data = data[i:i+chunk_samples]
        buf = io.BytesIO()
        sf.write(buf, chunk_data, sr, format='WAV')
        chunk_bytes = buf.getvalue()
        
        text, lat = _transcribe_chunk(chunk_bytes, lang_code)
        if text:
            transcripts.append(text)
        total_latency += lat
        
    return " ".join(transcripts), total_latency
"""

end_marker = 'raise RuntimeError(f"All {MAX_RETRIES+1} attempts failed: {last_err}")'
if end_marker not in content:
    print("could not find end marker")
    sys.exit(1)

parts = content.split(end_marker)
content = parts[0] + end_marker + "\n\n" + new_transcribe + parts[1]

with open('evaluate_prisma.py', 'w') as f:
    f.write(content)
print("patched")
