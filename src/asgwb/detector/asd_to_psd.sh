# !/bin/bash
# Set formatting equal to numpy's
# See https://numpy.org/doc/2.2/reference/generated/numpy.savetxt
for file in "$@"; do
    awk -v OFMT="%.18e" '{ print $1, $2**2 }' "$file" > "${file%.*}_psd.txt"
done
