#!/usr/bin/env python3
"""DeepStream PPE pipeline with a Python pad probe (pyds).  *** NOT YET RUN ON A GPU - verify on the demo box ***

  filesrc|v4l2src -> decode -> nvstreammux -> nvinfer(YOLOv8n PPE) -> nvtracker -> nvvideoconvert -> nvdsosd
      -> [probe: rule + red box + NO HELMET banner + counters + MQTT/JSONL events] -> encode -> mp4 (or display)

Usage (inside the DeepStream container, from repo root):
  python3 deepstream/ppe_deepstream.py --source samples/sample_h264.mp4 --out ds_output.mp4 \
      --events ds_events.jsonl --mqtt localhost:1883
  python3 deepstream/ppe_deepstream.py --source /dev/video0 --display          # USB camera, on-screen
Uses the same rule helpers as fallback/rules.py, so behaviour matches the (tested) ONNX Runtime fallback.
"""
import argparse, sys
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
import pyds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fallback"))
from events import EventSink
from rules import center_in

NAMES = (ROOT / "models" / "labels.txt").read_text().split()
CID = {n: i for i, n in enumerate(NAMES)}
UNTRACKED = 0xFFFFFFFFFFFFFFFF   # nvtracker's id for objects it has not (yet) tracked
STATE = {"streak": {}, "last_alert": {}, "persons": set(), "violators": set(), "alerts": 0, "sink": None, "args": None}


def osd_probe(pad, info, _):
    a, st = STATE["args"], STATE
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    lf = batch.frame_meta_list
    while lf:
        fm = pyds.NvDsFrameMeta.cast(lf.data)
        people, bad = [], []
        seen = set()
        lo = fm.obj_meta_list
        while lo:
            om = pyds.NvDsObjectMeta.cast(lo.data)
            r = om.rect_params
            box = (r.left, r.top, r.left + r.width, r.top + r.height)
            if om.class_id == CID["Person"]:
                people.append((om, box))
            elif om.class_id == CID["no_helmet"]:
                bad.append(box)
            lo = lo.next
        if fm.frame_num < 3:                           # diagnostic: what the probe actually receives
            ids = []; x = fm.obj_meta_list
            while x:
                o = pyds.NvDsObjectMeta.cast(x.data); ids.append((o.class_id, o.object_id)); x = x.next
            print(f"DIAG frame={fm.frame_num} objects={len(ids)} (class,object_id)={ids[:6]}", flush=True)
        now_bad = 0
        area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        helmetless = set()
        for b in bad:                                # each no_helmet box belongs to ONE person (tightest box)
            c = [(area(bx), i) for i, (_, bx) in enumerate(people) if center_in(b, bx)]
            if c:
                helmetless.add(min(c)[1])
        for i, (om, box) in enumerate(people):
            if om.object_id == UNTRACKED:
                continue                             # no stable identity yet: cannot attribute a streak
            tid = (fm.source_id, om.object_id)       # ids are per stream
            seen.add(tid)
            if tid not in st["persons"]:
                st["persons"].add(tid)
                if st["sink"]:
                    st["sink"].emit({"track_id": f"{tid[0]}-{tid[1]}", "label": "Person", "confidence": round(om.confidence, 3),
                                     "bbox": [], "frame": fm.frame_num})
            if i in helmetless:
                now_bad += 1
                st["streak"][tid] = st["streak"].get(tid, 0) + 1
                om.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                om.rect_params.border_width = 5
                last = st["last_alert"].get(tid)
                if st["streak"][tid] >= a.min_hits and (last is None or fm.frame_num - last >= a.cooldown):
                    st["last_alert"][tid] = fm.frame_num
                    st["violators"].add(tid); st["alerts"] += 1
                    if st["sink"]:
                        st["sink"].emit({"track_id": f"{tid[0]}-{tid[1]}", "label": "no_helmet", "confidence": round(om.confidence, 3),
                                         "bbox": [int(v) for v in box], "frame": fm.frame_num})
            else:
                st["streak"][tid] = 0
        for tid in list(st["streak"]):               # consecutive means consecutive: absent => reset
            if tid not in seen:
                st["streak"][tid] = 0
        dm = pyds.nvds_acquire_display_meta_from_pool(batch)
        dm.num_labels = 1
        t = dm.text_params[0]
        t.display_text = (f"NO HELMET x{now_bad}  |  " if now_bad else "") + \
                         f"persons {len(st['persons'])}  violators {len(st['violators'])}  alerts {st['alerts']}"
        t.x_offset, t.y_offset = 10, 10
        t.font_params.font_name, t.font_params.font_size = "Serif", 18
        t.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        t.set_bg_clr = 1
        t.text_bg_clr.set(1.0, 0.0, 0.0, 0.85) if now_bad else t.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
        pyds.nvds_add_display_meta_to_frame(fm, dm)
        lf = lf.next
    return Gst.PadProbeReturn.OK


def build(a):
    is_cam = a.source.startswith("/dev/video")
    src = (f"v4l2src device={a.source} ! video/x-raw,framerate=30/1 ! videoconvert ! nvvideoconvert ! "
           f"video/x-raw(memory:NVMM),format=NV12 ! m.sink_0 " if is_cam else
           f"filesrc location={a.source} ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 ")
    import platform
    sink = "nvegltransform ! nveglglessink sync=0" if platform.machine() == "aarch64" else "nveglglessink sync=0"
    tail = (sink if a.display else
            f"nvvideoconvert ! video/x-raw(memory:NVMM),format=I420 ! nvv4l2h264enc bitrate=4000000 ! h264parse ! "
            f"qtmux ! filesink location={a.out}")
    return Gst.parse_launch(
        src + f"nvstreammux name=m batch-size=1 width={a.width} height={a.height} live-source={int(is_cam)} "
        f"batched-push-timeout=40000 ! nvinfer config-file-path={a.config} ! "
        f"nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        f"ll-config-file={a.tracker} tracker-width=640 tracker-height=384 ! "
        f"nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! nvdsosd name=osd ! " + tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "samples" / "sample_h264.mp4"))
    ap.add_argument("--out", default="ds_output.mp4")
    ap.add_argument("--config", default=str(ROOT / "deepstream" / "config_infer_primary_yolov8_ppe.txt"))
    ap.add_argument("--tracker", default="/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/"
                                         "config_tracker_NvDCF_perf.yml")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--events"); ap.add_argument("--mqtt"); ap.add_argument("--topic", default="ppe/events")
    ap.add_argument("--min-hits", type=int, default=3); ap.add_argument("--cooldown", type=int, default=150)
    a = ap.parse_args()
    STATE["args"] = a
    if a.events or a.mqtt:
        STATE["sink"] = EventSink(a.events, a.mqtt, a.topic, "cam0")
    Gst.init(None)
    pipe = build(a)
    pipe.get_by_name("osd").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_probe, None)
    loop = GLib.MainLoop()
    bus = pipe.get_bus(); bus.add_signal_watch()

    def on_msg(_, m):
        if m.type == Gst.MessageType.EOS:
            loop.quit()
        elif m.type == Gst.MessageType.ERROR:
            print("ERROR:", m.parse_error()); loop.quit()
    bus.connect("message", on_msg)
    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        if not a.display:                            # let qtmux write the moov atom before tearing down
            pipe.send_event(Gst.Event.new_eos())
            bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        pipe.set_state(Gst.State.NULL)
        if STATE["sink"]:
            STATE["sink"].close()
    print(f"persons={len(STATE['persons'])} violators={len(STATE['violators'])} alerts={STATE['alerts']}")


if __name__ == "__main__":
    main()
