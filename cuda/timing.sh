#! /bin/bash

make clean
make
rm -f timings.csv

echo "Running timings..."
for i in `seq -f %.0f 10000000 10000000 100000000`; do
    ./sort.cx "$i" "0" "100000" >> timings.csv
done
