import runpod
import subprocess
import os
import torch


def handler(event):
    # Extract input arguments from the event payload
    args = event.get("input", {})
    audio_file = args.get("audio")
    model_name = args.get("whisper_model", "large-v2")
    batch_size = args.get("batch_size", 4)
    language = args.get("language", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stemming = args.get("stemming", True)
    suppress_numerals = args.get("suppress_numerals", False)

    if not audio_file:
        return {"error": "No audio file provided."}

    # Determine the target audio file after optional source separation
    if stemming:
        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_file}" -o temp_outputs --device "{device}"'
        )

        if return_code != 0:
            vocal_target = audio_file
        else:
            vocal_target = os.path.join(
                "temp_outputs",
                "htdemucs",
                os.path.splitext(os.path.basename(audio_file))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio_file

    # Run the diarization process
    process = subprocess.Popen([
        "python", "diarize_parallel.py", "-a", vocal_target,
        "--whisper-model", model_name,
        "--batch-size", str(batch_size),
        "--language", language if language else "",
        "--device", device,
        "--suppress_numerals" if suppress_numerals else ""
    ], stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    stdout, stderr = process.communicate()
    return_code = process.returncode

    if return_code != 0:
        return {"error": "Diarization failed.", "details": stderr.decode("utf-8")}

    return {"status": "success", "output": stdout.decode("utf-8")}


# Start RunPod worker
runpod.serverless.start({"handler": handler})
