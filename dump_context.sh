#!/bin/bash

# dump file contents to stdout, skipping commented lines
# usage: ./dump_context.sh file1 file2 ...
# bash dump_context.sh train_zf.py lsfpnet_encoding.py radial_lsfp.py mc.py ei.py transform.py utils.py eval.py lsp.py dataloader.py > context.txt

if [ $# -eq 0 ]; then
    echo "Error: No files specified" >&2
    echo "Usage: $0 file1 file2 ..." >&2
    exit 1
fi

for file in "$@"; do
    if [ -f "$file" ]; then
        echo "=== $file ==="
        # skip lines starting with # (ignoring leading whitespace) without empty lines
        grep -v '^\s*#' "$file" | grep -v '^\s*$'
        echo ""
    else
        echo "Error: File not found - $file" >&2
    fi
done