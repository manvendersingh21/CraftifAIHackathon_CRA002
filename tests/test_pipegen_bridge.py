"""PipeGen JSON output (layout documented in the PipeGen v0.5.1 lab, Step 10-3) -> detections for our rules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
from pipegen_bridge import dets_of, frames_of

LAB_LAYOUT = {"pipeline": "video_json_export", "source": "/home/lab/videos/people.mp4", "frames": [
    {"frame": 42, "width": 1920, "height": 1080,
     "objects": [{"label": "person", "confidence": 0.5591,
                  "bbox": {"left": 0.0, "top": 407.5, "width": 70.1, "height": 338.3}}],
     "shapes": []}]}


def test_frames_of_handles_file_single_frame_and_list():
    assert len(list(frames_of(LAB_LAYOUT))) == 1
    assert len(list(frames_of(LAB_LAYOUT["frames"][0]))) == 1
    assert len(list(frames_of([LAB_LAYOUT, LAB_LAYOUT]))) == 2
    assert list(frames_of({"something": "else"})) == []


def test_dets_of_converts_bbox_and_normalises_labels():
    d = dets_of(LAB_LAYOUT["frames"][0])
    assert d == [(0, 407, 70, 745, "Person", 0.5591)]
    assert dets_of({"objects": [{"label": "helmet", "score": .9, "bbox": [1, 2, 3, 4]}]}) == [(1, 2, 3, 4, "helmet", .9)]
    assert dets_of({"objects": [{"label": "x", "bbox": "bad"}]}) == []
