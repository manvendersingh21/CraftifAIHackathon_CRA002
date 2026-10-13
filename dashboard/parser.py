"""Schema-tolerant parser for DeepStream / PipeGen style JSON metadata.

Handles: nvmsgconv full schema ("object": {...}), minimal schema ("objects": ["id|l|t|r|b|label"]),
DeepStream-Yolo / custom probe output (lists of dicts with label/class/track_id ...), and
anything else (recursive walk). Unknown payloads still come back with raw text.
"""
import json, re, time

LABEL_KEYS = ("label", "obj_label", "class", "class_name", "classname", "classLabel", "class_label",
              "objectType", "object_type", "obj_class")
ID_KEYS = ("tracker_id", "track_id", "trackId", "trackingId", "object_id", "objectId", "obj_id", "id")
TS_KEYS = ("@timestamp", "timestamp", "ts", "time", "frame_ts", "ntp_timestamp")
SIZE_KEYS = ("frame_size", "frameSize")
FRAME_KEYS = ("frame_num", "frameNum", "frame_number", "frame", "frameId")
SRC_KEYS = ("sensorId", "sensor_id", "source_id", "sourceId", "camera_id", "cam", "source", "stream_id")
CONF_KEYS = ("confidence", "conf", "score", "probability")

VIOLATION_RE = re.compile(r"^(no[_\- ]?(helmet|hardhat|hard_hat|vest|goggle|goggles|gloves?|boots?|mask)"
                          r"|without[_\- ].+|violation)$", re.I)
PERSON_RE = re.compile(r"^(person|people|worker|human)$", re.I)


def _first(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = lower.get(k.lower())
        if v not in (None, "", []):
            return v
    return None


def _label_of(d):
    v = _first(d, LABEL_KEYS)
    if isinstance(v, (str, int, float)) and not isinstance(v, bool):
        return str(v)
    # nvmsgconv full schema: {"object": {"id": "3", "person": {...}}} -> nested key name as label
    return None


def _parse_minimal(s):
    # "id|left|top|right|bottom|label" (DeepStream minimal schema; some builds add conf)
    parts = s.split("|")
    if len(parts) >= 6:
        return {"id": parts[0], "label": parts[5], "bbox": parts[1:5],
                "conf": parts[6] if len(parts) > 6 else None}
    return None


def extract_objects(node, ctx=None, out=None, depth=0):
    """Recursively collect objects carrying a label. ctx carries inherited ts/frame/source."""
    if out is None:
        out = []
    ctx = dict(ctx or {})
    if depth > 12:
        return out
    if isinstance(node, dict):
        for name, keys in (("ts", TS_KEYS), ("frame", FRAME_KEYS), ("source", SRC_KEYS)):
            v = _first(node, keys)
            if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                ctx[name] = v
        if isinstance(node.get("sensor"), dict) and node["sensor"].get("id") is not None:
            ctx["source"] = node["sensor"]["id"]
        fs = _first(node, SIZE_KEYS)
        if isinstance(fs, (list, tuple)) and len(fs) == 2:
            ctx["size"] = [fs[0], fs[1]]
        label = _label_of(node)
        if label is not None and (_first(node, ID_KEYS) is not None or _first(node, CONF_KEYS) is not None
                                  or "bbox" in node or "rect_params" in node or VIOLATION_RE.match(label)
                                  or PERSON_RE.match(label)):
            bb = node.get("bbox")
            out.append({"label": label, "id": _first(node, ID_KEYS), "conf": _first(node, CONF_KEYS),
                        "ts": ctx.get("ts"), "frame": ctx.get("frame"), "source": ctx.get("source"),
                        "bbox": bb if isinstance(bb, list) and len(bb) == 4 else None, "size": ctx.get("size")})
        elif label is None and "object" in node and isinstance(node["object"], dict):
            o = node["object"]
            # full schema: label is the name of the nested sub-object (person/vehicle/...) or o["objectType"]
            sub = [k for k, v in o.items() if isinstance(v, dict) and k not in ("bbox", "location", "coordinate")]
            lab = _label_of(o) or (sub[0] if sub else None)
            if lab:
                bb = o.get("bbox")
                out.append({"label": lab, "id": _first(o, ID_KEYS), "conf": _first(o, CONF_KEYS),
                            "ts": ctx.get("ts"), "frame": ctx.get("frame"), "source": ctx.get("source"),
                            "bbox": bb if isinstance(bb, list) and len(bb) == 4 else None, "size": ctx.get("size")})
                return out
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                extract_objects(v, ctx, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, str):
                m = _parse_minimal(v)
                if m:
                    out.append({"label": m["label"], "id": m["id"], "conf": m["conf"], "ts": ctx.get("ts"),
                                "frame": ctx.get("frame"), "source": ctx.get("source"), "bbox": None, "size": None})
            else:
                extract_objects(v, ctx, out, depth + 1)
    return out


def parse(raw, topic=None):
    """Return {'raw','topic','objects','violations','json_ok','received'}."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        data = json.loads(raw)
        ok = True
    except Exception:
        data, ok = None, False
    objs = extract_objects(data) if ok else []
    if not ok:  # plain-text fallback: look for violation words
        for w in re.findall(r"\b(no[_\-]?\w+)\b", raw, re.I):
            if VIOLATION_RE.match(w):
                objs.append({"label": w, "id": None, "conf": None, "ts": None, "frame": None, "source": None,
                             "bbox": None, "size": None})
    for o in objs:
        o["violation"] = bool(VIOLATION_RE.match(o["label"]))
        o["person"] = bool(PERSON_RE.match(o["label"]))
        if o["id"] is not None:
            o["id"] = str(o["id"])
    return {"raw": raw[:4000], "topic": topic, "json_ok": ok, "objects": objs,
            "violations": [o for o in objs if o["violation"]], "received": time.time()}
