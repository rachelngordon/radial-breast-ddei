#!/usr/bin/env bash
#SBATCH -J torch_sanity
#SBATCH -p gpuq
##SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o slurm-%j.out

set -o pipefail
set -x                      # <-- echo each command for forensics
umask 077

# ---- EDIT THESE TWO LINES IF NEEDED ----
MICROMAMBA_SH="/gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/etc/profile.d/micromamba.sh"
ENV_NAME="recon_mri"
# ----------------------------------------

echo "=== HOST INFO ==="
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo

# Safety: verify micromamba hook exists
if [[ ! -f "$MICROMAMBA_SH" ]]; then
  echo "[FATAL] micromamba init script not found: $MICROMAMBA_SH"
  exit 3
fi

# Activate env
source "$MICROMAMBA_SH"
micromamba activate "$ENV_NAME" || { echo "[FATAL] micromamba activate failed"; exit 4; }

# Show Python/conda context immediately
which python || true
python -V || true
echo "CONDA_PREFIX=${CONDA_PREFIX:-<empty>}"
echo "MICROMAMBA_EXE=${MAMBA_EXE:-<unknown>}"
echo

# Minimal env hardening
unset LD_PRELOAD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
export PYTHONUNBUFFERED=1

OUT_DIR="${OUT_DIR:-$PWD/torch_sanity_${SLURM_JOB_ID:-manual}_$(hostname)}"
mkdir -p "$OUT_DIR"
echo "Logs will go to: $OUT_DIR"
echo

STATUS=0

echo "=== STEP A: Locate torch libs WITHOUT importing torch ==="
python - <<'PY' > "$OUT_DIR/libs.txt" || true
import sys, site, glob, os
paths=set()
for getter in (getattr(site,'getsitepackages',lambda:[]) , lambda:[site.getusersitepackages()]):
    for p in getter():
        if p: paths.add(p)
cands=[]
for p in paths:
    for pat in ("libtorch_cpu*.so","libc10*.so","libtorch.so"):
        cands += glob.glob(os.path.join(p,"torch*","lib",pat))
cands = sorted(set(cands))
print("\n".join(cands))
PY

echo "--- located libs ---"
if [ -s "$OUT_DIR/libs.txt" ]; then
  nl -ba "$OUT_DIR/libs.txt"
else
  echo "[FAIL] No torch libs found in site-packages." | tee -a "$OUT_DIR/summary.txt"
  STATUS=1
fi
echo

echo "=== STEP B: ldd on located libs ==="
> "$OUT_DIR/ldd_missing.txt"
if [ -s "$OUT_DIR/libs.txt" ]; then
  while IFS= read -r so; do
    [ -f "$so" ] || continue
    echo "---- LDD: $so ----" | tee -a "$OUT_DIR/ldd.txt"
    /usr/bin/ldd "$so" | tee -a "$OUT_DIR/ldd.txt"
    /usr/bin/ldd "$so" | awk '/not found/ {print FILENAME":"$0}' FILENAME="$so" >> "$OUT_DIR/ldd_missing.txt"
  done < "$OUT_DIR/libs.txt"
fi

if [ -s "$OUT_DIR/ldd_missing.txt" ]; then
  echo "[FAIL] Missing dependencies detected:" | tee -a "$OUT_DIR/summary.txt"
  cat "$OUT_DIR/ldd_missing.txt" | tee -a "$OUT_DIR/summary.txt"
  STATUS=1
elif [ -s "$OUT_DIR/libs.txt" ]; then
  echo "[OK] No missing deps reported by ldd." | tee -a "$OUT_DIR/summary.txt"
fi
echo

echo "=== STEP C: ctypes load test (no import torch) ==="
if [ -s "$OUT_DIR/libs.txt" ]; then
  python - <<'PY' "$OUT_DIR/libs.txt"
import ctypes, sys
libs=[l.strip() for l in open(sys.argv[1]) if l.strip()]
order = sorted(libs, key=lambda x: (0 if 'libc10' in x else 1 if 'libtorch_cpu' in x else 2, x))
ok=True
for lib in order:
    try:
        ctypes.CDLL(lib)
        print(f"[OK] ctypes loaded: {lib}")
    except Exception as e:
        print(f"[FAIL] ctypes load: {lib} -> {e!r}")
        ok=False
if not ok: raise SystemExit(2)
PY
  CTYPES_RC=$?
  if [ $CTYPES_RC -ne 0 ]; then
    echo "[FAIL] ctypes could not load one or more libs." | tee -a "$OUT_DIR/summary.txt"
    STATUS=1
  else
    echo "[OK] ctypes loaded all core libs." | tee -a "$OUT_DIR/summary.txt"
  fi
else
  echo "[SKIP] ctypes test (no libs discovered)." | tee -a "$OUT_DIR/summary.txt"
fi
echo

echo "=== STEP D: Hash libs (detect silent corruption) ==="
if [ -s "$OUT_DIR/libs.txt" ]; then
  sort -u "$OUT_DIR/libs.txt" | xargs -r -n 64 sha256sum > "$OUT_DIR/sha256.txt" || true
  head -n 20 "$OUT_DIR/sha256.txt" || true
  echo "[INFO] Full hashes at $OUT_DIR/sha256.txt"
else
  echo "[SKIP] hashing (no libs)."
fi
echo

echo "=== STEP E: CPU-only import torch ==="
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import sys, platform
print("PY:", sys.version.splitlines()[0])
print("PLATFORM:", platform.platform())
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("import torch (CPU-only) OK")
PY
RC=$?
if [ $RC -ne 0 ]; then
  echo "[FAIL] CPU-only 'import torch' crashed (SIGBUS/exception)." | tee -a "$OUT_DIR/summary.txt"
  STATUS=1
else
  echo "[OK] CPU-only 'import torch' succeeded." | tee -a "$OUT_DIR/summary.txt"
fi
echo

echo "=== STEP F: System context ==="
{
  echo "-- lscpu (model/flags) --"
  lscpu | egrep -i 'model name|flags|avx' || true
  echo
  echo "-- LD_LIBRARY_PATH --"
  echo "${LD_LIBRARY_PATH:-<empty>}"
  echo
  echo "-- CONDA_PREFIX --"
  echo "${CONDA_PREFIX:-<empty>}"
} | tee "$OUT_DIR/sys.txt"

echo
echo "=== SUMMARY ==="
cat "$OUT_DIR/summary.txt" 2>/dev/null || echo "[INFO] No issues recorded."
echo "Logs: $OUT_DIR"
exit $STATUS

