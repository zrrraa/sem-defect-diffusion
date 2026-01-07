#!/usr/bin/env bash
set -euo pipefail

DEF_DIR="/home/rzhong/project/sem-defect-diffusion/data/raw/defect"
MASK_DIR="/home/rzhong/project/sem-defect-diffusion/data/raw/mask"
CLEAN_DIR="/home/rzhong/project/sem-defect-diffusion/data/lama_inpaint"
OUT_FILE="annotations.jsonl"

# 清空输出文件
: > "$OUT_FILE"

find "$DEF_DIR" -type f -name "*.png" -print0 | while IFS= read -r -d '' defect_path; do
  fname="${defect_path##*/}"      # xxx.png
  id="${fname%.png}"              # xxx

  # 取缺陷图所属类别子目录：hole / particle / scratch
  parent_dir="${defect_path%/*}"  # .../defect/<cls>
  cls="${parent_dir##*/}"

  # 映射到 class_id
  case "$cls" in
    hole)     class_id=0 ;;
    particle) class_id=1 ;;
    scratch)  class_id=2 ;;
    *)
      echo "[WARN] unknown class folder: $cls (skip file=$defect_path)" >&2
      continue
      ;;
  esac

  # mask / clean 也在相同的子目录结构下
  mask_path="${MASK_DIR}/${cls}/${id}_mask.png"
  clean_path="${CLEAN_DIR}/${cls}/${id}.png"

  if [[ ! -f "$mask_path" ]]; then
    echo "[WARN] missing mask: $mask_path (skip id=$id, cls=$cls)" >&2
    continue
  fi
  if [[ ! -f "$clean_path" ]]; then
    echo "[WARN] missing clean: $clean_path (skip id=$id, cls=$cls)" >&2
    continue
  fi

  # 用 python 的 json.dumps 做安全转义
  python - "$id" "$defect_path" "$mask_path" "$clean_path" "$class_id" >> "$OUT_FILE" <<'PY'
import json, sys
id_, defect, mask, clean, class_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
obj = {
  "id": id_,
  "defect_path": defect,
  "mask_path": mask,
  "clean_path": clean,
  "class_id": class_id
}
print(json.dumps(obj, ensure_ascii=False))
PY

done

echo "[DONE] wrote: $OUT_FILE"
