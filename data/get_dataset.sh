#!/usr/bin/env bash
# Pulls Ultralytics "construction-ppe" (YOLO format, 11 classes incl. helmet/vest/no_helmet).
# Optional: SUBSET=N makes data/construction-ppe-subset with N train / N/4 val images for smoke tests.
set -euo pipefail
cd "$(dirname "$0")"
URL=https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
if [ ! -d construction-ppe ]; then
  curl -L --fail -o construction-ppe.zip "$URL"
  mkdir -p construction-ppe && unzip -q construction-ppe.zip -d construction-ppe && rm construction-ppe.zip
  # some zips nest a top-level folder
  [ -d construction-ppe/construction-ppe ] && mv construction-ppe/construction-ppe/* construction-ppe/ && rmdir construction-ppe/construction-ppe
fi
ROOT=$(pwd)/construction-ppe
NAMES='names: {0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none, 6: Person, 7: no_helmet, 8: no_goggle, 9: no_gloves, 10: no_boots}'
printf "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\n%s\n" "$ROOT" "$NAMES" > ppe.yaml
echo "Full dataset yaml: data/ppe.yaml"
if [ -n "${SUBSET:-}" ]; then
  S=$(pwd)/construction-ppe-subset; rm -rf "$S"
  for split in train val; do
    n=$SUBSET; [ $split = val ] && n=$(( SUBSET/4>0 ? SUBSET/4 : 1 ))
    mkdir -p "$S/images/$split" "$S/labels/$split"
    for img in $(ls "$ROOT/images/$split" | head -n $n); do
      cp "$ROOT/images/$split/$img" "$S/images/$split/"
      lbl="${img%.*}.txt"; [ -f "$ROOT/labels/$split/$lbl" ] && cp "$ROOT/labels/$split/$lbl" "$S/labels/$split/"
    done
  done
  printf "path: %s\ntrain: images/train\nval: images/val\n%s\n" "$S" "$NAMES" > ppe-subset.yaml
  echo "Subset yaml: data/ppe-subset.yaml"
fi
