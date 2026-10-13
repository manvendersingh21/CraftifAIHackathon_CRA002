import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fallback"))
from rules import PPEMonitor, IoUTracker, iou

P = lambda x, y, w=100, h=250, c=0.9: (x, y, x + w, y + h, "Person", c)
NH = lambda x, y: (x + 20, y, x + 80, y + 50, "no_helmet", 0.8)
H = lambda x, y: (x + 20, y, x + 80, y + 50, "helmet", 0.8)
V = lambda x, y: (x + 10, y + 60, x + 90, y + 160, "vest", 0.8)


def run(mon, frames):
    ev = []
    for i, dets in enumerate(frames):
        ev += mon.update(dets, i)
    return ev


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_tracker_keeps_id_when_person_moves():
    t = IoUTracker()
    ids = [t.update([(10 + 5 * i, 10, 110 + 5 * i, 260)], i)[0] for i in range(20)]
    assert len(set(ids)) == 1


def test_tracker_new_ids_for_two_people():
    t = IoUTracker()
    assert t.update([(0, 0, 100, 250), (400, 0, 500, 250)], 0) == [1, 2]


def test_helmeted_worker_no_event():
    assert run(PPEMonitor(absence=0), [[P(100, 100), H(100, 100)]] * 30) == []


def test_one_event_per_track_not_per_frame():
    ev = run(PPEMonitor(absence=0), [[P(100, 100), NH(100, 100)]] * 100)
    assert len(ev) == 1 and ev[0]["label"] == "no_helmet" and ev[0]["track_id"] == 1


def test_realerts_after_cooldown():
    ev = run(PPEMonitor(cooldown=50, absence=0), [[P(100, 100), NH(100, 100)]] * 120)
    assert len(ev) == 3  # frames 2, 52, 102


def test_one_frame_flicker_suppressed():
    frames = [[P(100, 100), H(100, 100)]] * 5 + [[P(100, 100), NH(100, 100)]] + [[P(100, 100), H(100, 100)]] * 5
    assert run(PPEMonitor(absence=0), frames) == []


def test_two_people_only_violator_flagged():
    mon = PPEMonitor(absence=0)
    ev = run(mon, [[P(50, 100), H(50, 100), P(500, 100), NH(500, 100)]] * 10)
    assert len(ev) == 1 and ev[0]["bbox"][0] == 500
    assert len(mon.persons) == 2 and len(mon.violators) == 1


def test_no_helmet_box_without_person_still_alerts():
    ev = run(PPEMonitor(absence=0), [[(300, 100, 380, 160, "no_helmet", 0.7)]] * 10)
    assert len(ev) == 1


def test_vest_rule_off_by_default_on_when_requested():
    frames = [[P(100, 100), H(100, 100)]] * 10
    assert run(PPEMonitor(absence=0), frames) == []
    ev = run(PPEMonitor(require_vest=True, absence=0), frames)
    assert [e["label"] for e in ev] == ["no_vest"]
    assert run(PPEMonitor(require_vest=True, absence=0), [[P(100, 100), H(100, 100), V(100, 100)]] * 10) == []


def test_track_survives_short_occlusion():
    mon = PPEMonitor(absence=0)
    frames = [[P(100, 100), NH(100, 100)]] * 5 + [[]] * 10 + [[P(104, 100), NH(104, 100)]] * 5
    ev = run(mon, frames)
    assert len(ev) == 1 and len(mon.persons) == 1


def test_new_persons_reported_once():
    mon = PPEMonitor(absence=0)
    mon.update([P(100, 100)], 0)
    assert [t for t, _ in mon.new_persons] == [1]
    mon.update([P(102, 100)], 1)
    assert mon.new_persons == []


def test_one_no_helmet_box_credited_to_one_person_only():
    A, B = (100, 100, 200, 400, "Person", .9), (150, 100, 260, 400, "Person", .9)
    nh = (110, 100, 190, 160, "no_helmet", .8)
    ev = run(PPEMonitor(absence=0), [[A, B, nh]] * 10)
    assert len(ev) == 1


def test_person_detection_flicker_does_not_create_new_id():
    mon = PPEMonitor(absence=0)
    frames = [[P(100, 100), NH(100, 100)]] * 3 + [[NH(100, 100)]] * 6 + [[P(100, 100), NH(100, 100)]] * 5
    ev = run(mon, frames)
    assert len(ev) == 1 and len(mon.persons) == 1


def test_streak_must_be_consecutive_across_missed_frames():
    mon = PPEMonitor(min_hits=3, absence=0)
    frames = [[P(100, 100), NH(100, 100)]] * 2 + [[P(100, 100), H(100, 100)]] * 8 + [[P(100, 100), NH(100, 100)]] * 2
    assert run(mon, frames) == []


def test_orphan_box_counts_as_person_only_after_alert():
    mon = PPEMonitor(absence=0)
    mon.update([(300, 100, 380, 160, "no_helmet", .7)], 0)
    assert mon.persons == set()


def test_absence_rule_flags_person_who_never_wears_helmet():
    ev = run(PPEMonitor(), [[P(100, 100)]] * 60)
    assert len(ev) == 1 and ev[0]["label"] == "no_helmet"


def test_absence_rule_quiet_when_helmet_seen_regularly():
    frames = [[P(100, 100), H(100, 100)] if i % 5 == 0 else [P(100, 100)] for i in range(80)]
    assert run(PPEMonitor(), frames) == []


def test_absence_rule_ignores_tiny_far_people():
    assert run(PPEMonitor(), [[(100, 100, 120, 150, "Person", .9)]] * 60) == []


def test_helmet_in_lower_half_does_not_count():
    feet_hat = (110, 330, 190, 390, "helmet", .9)           # on the legs of a 250px-tall person
    assert len(run(PPEMonitor(), [[P(100, 100), feet_hat]] * 60)) == 1


def test_helmeted_worker_never_flagged_over_long_run():
    assert run(PPEMonitor(), [[P(100, 100), H(100, 100)]] * 200) == []


def test_absence_rule_skips_person_cut_off_at_top_edge():
    assert run(PPEMonitor(), [[(100, 0, 200, 250, "Person", .9)]] * 60) == []   # head out of frame: cannot judge


def test_v2_head_class_counts_as_bare_head():
    head = lambda x, y: (x + 20, y, x + 80, y + 50, "head", 0.8)
    ev = run(PPEMonitor(absence=0), [[P(100, 100), head(100, 100)]] * 10)
    assert len(ev) == 1 and ev[0]["label"] == "no_helmet"
