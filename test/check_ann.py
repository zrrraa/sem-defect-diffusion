from __future__ import print_function
import json
import os
import cv2

ann = "data/raw/annotations.jsonl"

total = 0
ok_read = 0
ok_shape = 0
ok_mask = 0
bad = []

with open(ann, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        a = json.loads(line)
        total += 1

        dp, cp, mp = a["defect_path"], a["clean_path"], a["mask_path"]
        if not (os.path.exists(dp) and os.path.exists(cp) and os.path.exists(mp)):
            bad.append((a["id"], "missing_file"))
            continue

        i_def = cv2.imread(dp, cv2.IMREAD_GRAYSCALE)
        i_cln = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
        m_def = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if i_def is None or i_cln is None or m_def is None:
            bad.append((a["id"], "imread_none"))
            continue
        ok_read += 1

        if i_def.shape != i_cln.shape or i_def.shape != m_def.shape:
            bad.append((a["id"], "shape_mismatch"))
            continue
        ok_shape += 1

        if (m_def > 0).sum() <= 0:
            bad.append((a["id"], "empty_mask"))
            continue
        ok_mask += 1

print("total:", total)
print("ok_read:", ok_read)
print("ok_shape:", ok_shape)
print("ok_mask:", ok_mask)
print("bad:", len(bad))
print("first20_bad:", bad[:20])
