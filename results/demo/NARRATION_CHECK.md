# Narration vs screen check — full_demo_narrated.mp4

Note: the s1–s5 text was not included with the task, so the lines below were recovered by
transcribing `voice/s1..s5.wav` (vosk small-en model); minor ASR garble corrected against
on-screen text (e.g. "y willow v eight" → YOLOv8n, "cornell" → CoreML). Frames were extracted
with ffmpeg (no edits to the video) to `check_<t>.png` and verified via OCR (tesseract) plus
red/green bounding-box pixel analysis.

Cues: 0 s, 13 s, 28 s, 43 s, 53 s (extra frame at 70 s).

| Cue | Narration (recovered) | Verdict |
|---|---|---|
| 0 s (s1) | "Workplace Safety Sentinel… construction sites lose workers to head injuries every year; missing hard hats are a leading cause; watching every camera by hand does not scale." | MATCH — screen shows the "Workplace Safety Sentinel / Edge PPE compliance — hard-hat detection" dashboard over a construction-site clip with detections already running (red NO HELMET boxes, incident feed active). |
| 13 s (s2) | "Runs on the edge; detector is a YOLOv8 nano fine-tuned on construction PPE, running live on this laptop with CoreML only; small event messages leave the device, never video." | MATCH — status bar reads "CoreML + ppe_yolov8n.onnx + local" at 22.3 FPS (edge/laptop claim on screen); "metadata/evidence only, never raw video" is the dashboard's documented design (index.html lead) though that exact sentence is not visible at this moment. |
| 28 s (s3) | "Real site clip, licensed footage from Wikimedia Commons; boxes are people/PPE, red means no hard hat, each violator counted once by tracker id." | MATCH — attribution "via Wikimedia Commons · CC BY-SA 4.0 — source" is on screen; pixel analysis confirms red violation bboxes + green OK boxes; UNIQUE VIOLATORS counter and track-id'd incident cards (track 13, track 4) are shown. |
| 43 s (s4) | "Incident cards appear with evidence photos as violations are confirmed; counters and timeline update live from the detector." | MATCH — incident feed shows NO HELMET cards (track 25/16/13 with conf % and timestamps, evidence thumbnails per dashboard design); counters changed since 28 s (PERSONS 145→160, later 165) and the incidents/min timeline panel is live. |
| 53 s (s5) | "Accuracy reported honestly: mAP@0.5 ≈ 0.57 overall; false alarms and missed bare heads documented per clip in the review; thank you." | MISMATCH — the screen keeps showing the dashboard (compliance 70.3%, incident feed); no mAP number or review table is visible at 53 s (nor at 60/70 s). The claims are true of the repo (SUBMISSION.filled.md: mAP50 0.571; results/real_videos/REVIEW.md per-clip false alarms/misses) but the narration describes off-screen documents. |
| 70 s (extra) | (narration ended ~66 s) | Screen in steady end state: dashboard idle, green OK boxes dominate, counters stable — consistent with the closing line. |

Summary: 4/5 cues match the screen. The only mismatch is s5 (53 s): the accuracy/review claims are
real but never shown on screen; if strict A/V sync is wanted, add an overlay (mAP + review table)
or re-record that segment, otherwise the voiceover works as a closing reference to the written docs.
