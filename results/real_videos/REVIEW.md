# Real-video review: live pipeline on 27 online clips

Run: `.venv/bin/python scripts/eval_videos.py --manifest <scratch manifest of every data/videos/*.mp4> --stride 2 --out results/real_videos`
Detector: CoreML `ppe_yolov8n.onnx`, median 8-17 ms/frame. Raw numbers are in `eval.json`, the frames are in `<clip>_sheet.jpg`.

**How I judged:** I opened all 26 contact sheets by eye. Where a sheet was too small to read a head, I pulled the full-resolution source frame at the same timestamp and cropped it. I checked about 20 frames this way, across excavator, both Malta clips, Manila, night, crane_far_a, bridge, rebar, road_work, scaffold, asphalt and concrete. Counts are **person-instances on the 12 panels**, not unique people. Columns:
- **OK alert** = NO HELMET on a head that really has no hard hat (bare, cap, beanie or hood).
- **False alarm** = NO HELMET on someone wearing a hard hat, or on a non-person.
- **Missed violation** = a bare or capped head labelled `OK`.
- **Untracked** = a clearly visible person with no track box at all.

Counts with `~` come from far or small people, so they are approximate. Nothing here is ground truth.

| Clip | Scene | People | OK alert | False alarm | Missed violation | Untracked | Failure modes seen |
|---|---|---|---|---|---|---|---|
| mk_phone_call_yellow_helmet | close-up, yellow helmet, phone | 1 | 0 | 0 | 0 | ~10 | Extreme close-up: helmet and vest detected but no Person box, so no track |
| mk_two_builders_close | two bricklayers, yellow/orange helmets | 2 | 0 | 0 | 0 | 12 | Orange-helmet worker (back view) never tracked. Yellow one is OK on all 12 |
| mk_warehouse_white_helmet | n/a | n/a | n/a | n/a | n/a | n/a | **Could not open** ("moov atom not found"). The file looks truncated or still downloading |
| wm_abseil_tower_block | abseiler on tower block facade | 1 | 0 | 0 | 0 | 0 | OK on all panels (orange helmet). OK boxes on a chimney and a lamppost. eval.json logs 4 alerts that are not on the panels |
| wm_asphalt_crew_street | street asphalt crew, cold tint | 4-6 | 2 | 0 | 2 | ~45 | Far/small: after t≈4 s nobody is tracked. All workers are bare-headed, but #1 is OK. "helmet" fires on a road sign |
| wm_bamboo_scaffold_hk_a | HK bamboo scaffold, one worker | 1 | 0 | 2 | 0 | 0 | Yellow helmet worker flagged when bent over (helmet low and partly hidden) |
| wm_bamboo_scaffold_hk_b | two climbers, bamboo scaffold | 2 | 0 | ~1 | 0 | ~6 | Blue hard hats read as OK. Far end of the clip: one FA and lost tracks |
| wm_bridge_site_far | bridge repair, far, snow | 2 | 0 | 3 | 0 | ~20 | **Excavator bucket tracked as a person and flagged**. OK boxes on the red lift. Real workers too small |
| wm_concrete_pour_far | concrete pour, Tyvek suits | 4-6 | ~6 | ~1 | ~3 | ~6 | Beanie/hood wearers correctly flagged (good). Some beanies still OK. White helmets correctly OK |
| wm_construction_crane_far_a | site seen over fence, far | 4-8 | 0 | 4 (+1 unclear) | 0 | ~30 | **White helmet flagged** (#1). **Hi-vis jacket hanging on formwork flagged as person** (#3). Far people |
| wm_construction_far_b | precast panel lift, far | 2-6 | 0 | 0 | 0 | ~30 | Mostly sky/panel shots. Tiny workers untracked. eval.json's 5 alerts are not on the panels |
| wm_crane_crew_close | crane riggers, close | 4-5 | 0 | 4 | 0 | ~4 | Crouching worker (#9) occluded by a colleague: flagged although his helmet shows. It switches to OK once visible |
| wm_excavator_operator | mini-excavator, operator bare-headed | 1 (+3 bg) | 11 | 0 | 1 | ~10 | **Correct sustained alert** (verified on full-res frame). Breaker attachment tracked as an OK "person". Background workers are small |
| wm_hoist_cage_far | hoist cage, white helmets, far | 2-4 | 0 | 0 | 0 | ~16 | White helmets correctly OK. Tracks dropped in the second half of the clip |
| wm_indoor_scaffold_lowres | indoor scaffold, backlit, low-res | 3-4 | 0 | 0 | 0 | ~42 | **Zero people detected in the whole clip** (backlight, dark, low-res) |
| wm_malta_site_high_angle | square works, high angle | 4-8 | ~10 | 0 | ~12 | ~10 | Baseball cap (#15), beanie (#18), green cap (operator #11) and hood all correctly flagged. Many other bare heads OK |
| wm_malta_site_high_angle_b | tree planting crew, high angle | 8-12 | ~13 | ~5 | ~40 | few | Full-res: **almost no one wears a helmet**, yet most are OK (no helmet evidence, still OK). White bulk bag tracked and flagged (#13/#35) |
| wm_manhole_high_angle | top-down manhole, two workers | 2 | 0 | 2 | 0 | 24 | Top-down view: the two real workers are never detected. A huge box over bench/rubble gets OK, then NO HELMET |
| wm_manila_subway_visit | VIP site visit, white helmets | 6-15 | ~8 | ~1 | ~10 | ~15 | Bare-headed officials flagged correctly in the close shots. Many bare heads OK in crowd shots. Some shots have no people |
| wm_night_roadwork_jp | night roadwork, floodlight | 3-4 | 0 | 7 | 0 | 2 | **Night/overexposed white helmet, head bowed** over jackhammer: flagged from t=3.3 s on |
| wm_paving_worker_ru | paver laying bricks, cloth cap | 1 | 0 | 0 | 9 | 3 | **Cloth cap = no hard hat, labelled OK** in all tracked panels. Partly behind a fence |
| wm_prefab_house_assembly | prefab house, far, old footage | ~8 | 0 | 0 | 0 | ~70 | Far/small, low quality: almost nobody is tracked |
| wm_rebar_cage_workers | rebar cage, white helmets | 3-5 | 0 | 2 | 0 | ~10 | White helmet, crouched head-down (#16) flagged. Otherwise solid OKs |
| wm_road_work_us_lowres | chip-seal road crew, 320x240 | 1-3 | 0 | ~4 | 0 | 0 | 320x240 source. Green and tan hard hats flagged. OK boxes on trucks |
| wm_scaffold_dismantling_fr | Paris courtyard scaffold, 2 workers | 2 | 9 | 0 | 2 | 1 | **Beanie worker (#3) flagged, white-helmet worker (#1) OK**: clean contrast |
| wm_skyway_inspection_ph | VIP skyway inspection, crowd | 5-15 | ~20 | ~1 | ~15 | few | Civilians mostly bare-headed: many correct flags. The model's `none` class fires on bare heads, but rules.py ignores it, so some stay OK |
| wm_welder_balcony_india | welder on balcony, no helmet | 1 | 7 | 0 | 1 | 4 | **Correct sustained alert** on a bare-headed welder with sparks. Takes ~2.7 s to confirm |

## Overall (26 readable clips, 12 panels each, approximate)

- **Correct NO HELMET:** about 85 person-instances, concentrated in 8 clips: excavator, welder, scaffold_dismantling, both Malta clips, Manila, skyway and concrete_pour.
- **False alarms:** about 35 across 10 clips. About 13 are hard-hat wearers with head bowed, occluded, at night or under a white helmet. About 10 are on non-persons (excavator bucket, hanging jacket, bulk bag, manhole rubble box). About 5 are coloured hard hats in low-res footage.
- **Missed violations (bare/cap/beanie head shown OK):** about 95. Most are in Malta B (≈40), Manila and skyway crowds, and paving_ru (cloth cap).
- **Untracked visible people:** about 370. Far/small people dominate, plus indoor_scaffold, where nothing is detected.
- The rate of correct versus false alarms is acceptable on close, well-lit clips. The dominant error is **under-alerting**: a person gets OK whenever the `no_helmet` class does not fire for `absence` frames *and* a weak `helmet` hit appears anywhere in the box.

**Top 3 failure modes**
1. **Far or small people are not detected or tracked:** prefab, indoor_scaffold, asphalt, bridge, construction_far_b, crane_far_a, hoist, manhole. This accounts for most misses.
2. **Bare heads, caps, beanies and hoods labelled OK** (missed violations): Malta B, paving_ru, Manila and skyway crowds, asphalt. The model's `none` class often fires on these heads, but rules.py does not use it.
3. **False alarms on helmeted workers whose helmet is hard to see** (head bowed, occluded, night glare, white helmet) **and on non-person objects** (excavator bucket, hanging jacket, bulk bag, rubble). Seen in night_roadwork, crane_crew_close, rebar, bamboo_a, crane_far_a, bridge, Malta B and manhole.

## Best clips for a live demo
1. **wm_scaffold_dismantling_fr**: two workers side by side. The white helmet stays OK and the beanie is flagged NO HELMET on 9/12 panels, with no false alarms. This is the clearest correct-vs-violation contrast.
2. **wm_welder_balcony_india**: one bare-headed welder with sparks, flagged continuously from t≈5 s, with no false alarms.
3. **wm_excavator_operator**: the bare-headed operator is flagged on 11/12 panels (verified on a full-res frame). Background helmeted workers show OK.

Runner-up: **wm_concrete_pour_far** (t≥21 s: white-helmet worker OK next to flagged beanie wearers). Avoid night_roadwork, crane_crew_close and road_work_us_lowres in a demo, because they show visible false alarms on helmeted workers.

## Stride check (`results/real_videos/stride1/`)
I reran the 3 clips with the most false alarms at `--stride 1`: night_roadwork, crane_crew_close and road_work_us_lowres.
- Results were nearly the same as stride 2. Violators: 3/3/5 at stride 1 versus 3/3/5. Viol-frame %: 62.5/22.3/22.9 versus 63.6/28.0/26.8.
- The same false alarms show up on the stride-1 sheets: night #1 bowed white helmet, crane #9 occluded, road #1 green hard hat. **Stride did not cause the errors.**

## Harness bug (not fixed, code not edited)
`scripts/eval_videos.py` picks sheet frames with `if any(abs(idx - p) < a.stride for p in picks) and len(sheet) < 12`.
- At stride 2, a pick at an odd frame matches both idx-1 and idx+1, so the same moment is added twice (e.g. t=5.4/5.5, 21.8/21.8).
- The 12 slots then fill before the last picks are reached, so stride-2 sheets **stop at ~85% of the clip and contain near-duplicate panels**. Example: road_work_us ends at t=38.8 s at stride 2 but t=53.4 s at stride 1.
- Fix: consume each pick once, e.g.
  `hit = next((p for p in picks if abs(idx - p) < a.stride), None)`
  `if hit is not None: picks.discard(hit); <draw panel>`
  The same issue makes stride-2 sheets under-sample the end of every clip.
