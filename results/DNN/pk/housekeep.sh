#!/bin/bash

# Navigate to 10_10_10 and move files
cd 10_10_10 || exit
mv *100* ../10_10_100

# Navigate to 50_50_10 and move files
cd ../50_50_10 || exit
mv *100* ../50_50_100

# Navigate to 100_100_10 and move files
cd ../100_100_10 || exit
mv *100.* ../100_100_100
