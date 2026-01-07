#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-./data/lama_inpaint/hole}"
RADIUS="${2:-16}"

# dialte masks, only deal with *_mask.png files
find "$DIR" -type f -name "*_mask.png" -print0 | \
while IFS= read -r -d '' f; do
  tmp="$(mktemp "${TMPDIR:-/tmp}/maskXXXXXX.png")"
  echo "[INFO] dilate: $f  (radius=$RADIUS)"
  python ./src/dilate_mask.py "$f" "$tmp" --radius "$RADIUS"
  mv -f "$tmp" "$f"
done

echo "[DONE] All masks processed in: $DIR"

# # inpainting with LaMa
# cd third_party/lama
# export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
# python3 bin/predict.py model.path=/home/share/rzhong/model/big-lama indir=/home/rzhong/project/sem-defect-diffusion/data/lama_inpaint outdir=$(pwd)/output
# cd ../../

# # create annotations.jsonl
# DEF_DIR="/home/rzhong/project/sem-defect-diffusion/data/raw/defect"
# MASK_DIR="/home/rzhong/project/sem-defect-diffusion/data/raw/mask"
# CLEAN_DIR="/home/rzhong/project/sem-defect-diffusion/data/lama_inpaint"
# OUT_FILE="./data/raw/annotations.jsonl"
# CLASS_ID=0

# : > "$OUT_FILE"

# find "$DEF_DIR" -type f -name "*.png" -print0 | while IFS= read -r -d '' defect_path; do
#   fname="${defect_path##*/}"
#   id="${fname%.png}"

#   mask_path="${MASK_DIR}/${id}_mask.png"
#   clean_path="${CLEAN_DIR}/${id}.png"

#   if [[ ! -f "$mask_path" ]]; then
#     echo "[WARN] missing mask: $mask_path (skip id=$id)" >&2
#     continue
#   fi
#   if [[ ! -f "$clean_path" ]]; then
#     echo "[WARN] missing clean: $clean_path (skip id=$id)" >&2
#     continue
#   fi

#   python - "$id" "$defect_path" "$mask_path" "$clean_path" "$CLASS_ID" >> "$OUT_FILE" <<'PY'
# import json, sys
# id_, defect, mask, clean, class_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

# # 输出路径按你示例写成相对路径（保持原样）
# obj = {
#   "id": id_,
#   "defect_path": defect,
#   "mask_path": mask,
#   "clean_path": clean,
#   "class_id": class_id
# }
# print(json.dumps(obj, ensure_ascii=False))
# PY

# done

# echo "[DONE] wrote: $OUT_FILE"