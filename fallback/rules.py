"""PPE rules + lightweight tracking, independent of the inference backend (pure Python, no cv2/onnx).

Detections are (x1, y1, x2, y2, class_name, score) tuples. `PPEMonitor.update()` tracks people across frames,
decides per person whether a violation holds, and returns events de-duplicated per track id.
"""
from dataclasses import dataclass, field

PERSON = "Person"
HELMET, NO_HELMET = "helmet", "no_helmet"
VEST = "vest"


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside_frac(inner, outer):
    """Fraction of `inner` box area that lies inside `outer`."""
    ix1, iy1, ix2, iy2 = max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1) / area if area > 0 else 0.0


def center_in(inner, outer):
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


@dataclass
class Track:
    id: int
    box: tuple
    last_frame: int
    hits: int = 1
    violating: set = field(default_factory=set)       # labels currently violated
    streak: dict = field(default_factory=dict)        # label -> (consecutive count, last frame)
    last_alert: dict = field(default_factory=dict)    # label -> frame of last emitted event
    last_helmet: int = -10**9                          # last frame a helmet box sat on this person's head


class IoUTracker:
    """Greedy IoU association (good enough for a fixed camera; DeepStream uses nvtracker instead)."""

    def __init__(self, iou_thr=0.3, max_age=30):
        self.iou_thr, self.max_age, self.tracks, self._next = iou_thr, max_age, {}, 1

    def update(self, boxes, frame):
        pairs = sorted(((iou(t.box, b), tid, bi) for tid, t in self.tracks.items() for bi, b in enumerate(boxes)),
                       reverse=True)
        used_t, used_b, assign = set(), set(), {}
        for s, tid, bi in pairs:
            if s < self.iou_thr:
                break
            if tid in used_t or bi in used_b:
                continue
            used_t.add(tid); used_b.add(bi); assign[bi] = tid
        for bi, b in enumerate(boxes):
            if bi not in assign:
                assign[bi] = self._next
                self.tracks[self._next] = Track(self._next, tuple(b), frame)
                self._next += 1
            else:
                t = self.tracks[assign[bi]]
                t.box, t.last_frame, t.hits = tuple(b), frame, t.hits + 1
        for tid in [i for i, t in self.tracks.items() if frame - t.last_frame > self.max_age]:
            del self.tracks[tid]
        return [assign[i] for i in range(len(boxes))]


LOOSE_BASE = 1_000_000   # id namespace for no_helmet boxes that no person detection explains


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


class PPEMonitor:
    """Per-person PPE rule.

    * Each `no_helmet` box is credited to ONE person: the smallest person box containing its center.
    * If that person is not detected this frame but a live track contains the box, the track coasts (no new id).
    * A `no_helmet` box that no live person explains gets its own loose track (id >= LOOSE_BASE); it joins the
      headcount only once it alerts.
    * Helmet-absence rule (the explicit no_helmet class is a weak detector, `helmet`/`Person` are strong): a person
      tracked for >= `absence` frames and taller than `min_person_h` px who has had NO `helmet` box on the upper half
      of their box for `absence` frames is a no_helmet violation. Any helmet sighting resets it, so detector
      flicker does not cause alerts. Persons whose box touches the top frame edge are skipped (head out of view).
    * With require_vest=True a person with no `vest` box (each vest is credited to the person it sits in most)
      violates `no_vest`.
    * A violation must hold `min_hits` CONSECUTIVE frames before alerting, and a track re-alerts the same label only
      after `cooldown` frames.
    """

    def __init__(self, require_vest=False, min_hits=3, cooldown=150, iou_thr=0.3, max_age=30, absence=15,
                 min_person_h=60, edge_margin=3):
        self.require_vest, self.min_hits, self.cooldown = require_vest, min_hits, cooldown
        self.absence, self.min_person_h = absence, min_person_h   # helmet-absence rule (absence=0 disables)
        self.edge_margin = edge_margin                            # box touching the top edge => head out of view
        self.tracker = IoUTracker(iou_thr, max_age)        # people
        self.loose = IoUTracker(iou_thr, max_age)          # orphan no_helmet boxes
        self.violators, self.persons = set(), set()        # unique ids
        self.new_persons = []                              # (id, conf) first counted in the last update()

    def _check(self, tr, tid, labels, conf, box, frame, out):
        for lab in labels:
            c, lf = tr.streak.get(lab, (0, -2))
            c = c + 1 if lf == frame - 1 else 1
            tr.streak[lab] = (c, frame)
            if c < self.min_hits:
                continue
            last = tr.last_alert.get(lab)
            if last is None or frame - last >= self.cooldown:
                tr.last_alert[lab] = frame
                self.violators.add(tid)
                if tid not in self.persons:
                    self.persons.add(tid); self.new_persons.append((tid, float(conf)))
                out.append({"track_id": tid, "label": lab, "confidence": round(float(conf), 3),
                            "bbox": [int(v) for v in box], "frame": frame})

    def update(self, dets, frame):
        people = [d for d in dets if d[4] == PERSON]
        bad = [d for d in dets if d[4] == NO_HELMET]
        vests = [d for d in dets if d[4] == VEST]
        ids = self.tracker.update([p[:4] for p in people], frame)
        out, self.new_persons = [], []
        for p, tid in zip(people, ids):
            if tid not in self.persons:
                self.persons.add(tid); self.new_persons.append((tid, float(p[5])))
        # --- ownership of no_helmet boxes: smallest containing person; else a coasting live track; else loose
        bad_conf, orphans = {}, []
        for b in bad:
            c = [(_area(p), tid) for p, tid in zip(people, ids) if center_in(b, p)]
            if c:
                tid = min(c)[1]
            else:
                live = [(_area(t.box), t.id) for t in self.tracker.tracks.values()
                        if t.last_frame != frame and center_in(b, t.box)]
                tid = min(live)[1] if live else None
            if tid is None:
                orphans.append(b)
            else:
                bad_conf[tid] = max(bad_conf.get(tid, 0.0), b[5])
        for h in (d for d in dets if d[4] == "helmet"):     # a helmet on a head: smallest person whose upper half holds it
            c = [(_area(p), tid) for p, tid in zip(people, ids)
                 if center_in(h, p) and (h[1] + h[3]) / 2 <= p[1] + 0.5 * (p[3] - p[1])]
            if c:
                self.tracker.tracks[min(c)[1]].last_helmet = frame
        vested = set()
        for v in vests:
            best = max(((inside_frac(v, p), tid) for p, tid in zip(people, ids)), default=(0, None))
            if best[0] > 0.5:
                vested.add(best[1])
        for p, tid in zip(people, ids):
            tr = self.tracker.tracks[tid]
            labels = {NO_HELMET} if tid in bad_conf else set()
            if (self.absence and tr.hits >= self.absence and p[3] - p[1] >= self.min_person_h
                    and p[1] > self.edge_margin and frame - tr.last_helmet > self.absence):
                labels.add(NO_HELMET)
            if self.require_vest and tid not in vested:
                labels.add("no_vest")
            tr.violating = labels
            self._check(tr, tid, labels, bad_conf.get(tid, p[5]), p[:4], frame, out)
        for tid, conf in bad_conf.items():                 # coasting tracks (person not detected this frame)
            tr = self.tracker.tracks[tid]
            if tr.last_frame != frame:
                tr.violating = {NO_HELMET}
                self._check(tr, tid, {NO_HELMET}, conf, tr.box, frame, out)
        lids = self.loose.update([b[:4] for b in orphans], frame)
        for b, lid in zip(orphans, lids):
            tr = self.loose.tracks[lid]
            tr.violating = {NO_HELMET}
            self._check(tr, LOOSE_BASE + lid, {NO_HELMET}, b[5], b[:4], frame, out)
        return out

    def box_of(self, tid):
        t = self.loose.tracks.get(tid - LOOSE_BASE) if tid >= LOOSE_BASE else self.tracker.tracks.get(tid)
        return t.box if t else None

    def annotations(self, frame):
        """(box, track_id, violating_labels) for tracks matched in `frame`, for drawing."""
        return ([(t.box, t.id, set(t.violating)) for t in self.tracker.tracks.values() if t.last_frame == frame] +
                [(t.box, LOOSE_BASE + t.id, set(t.violating)) for t in self.loose.tracks.values()
                 if t.last_frame == frame])
