#!/bin/bash

# Default value for b (batch size)
b=4
outfile=""

# Parse the named argument for b
while [ $# -gt 0 ]; do
    case "$1" in 
        --b=*)
            b="${1#*=}"
        ;;
        --outfile=*)
            outfile="${1#*=}"
        ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Check if outfile is set
if [ -z "$outfile" ]; then
    echo "outfile not set"
    exit 1
fi

echo "Profiling the kernel using the batch size: $b"

ncu --target-processes=all --set full --export $outfile --page "details" --import-source yes --call-stack python tests/benchmark/run_kernel_axis1.py --b=$b