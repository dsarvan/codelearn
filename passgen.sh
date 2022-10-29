#!/usr/bin/env bash
# passgen.sh
# Script to generate 12 character passwords

echo 'Generating 12-character passwords'

for ((n=0;n<12;n++))
do dd if=/dev/urandom count=1 2> /dev/null | uuencode -m - | sed -ne 2p | cut -c-12
done
