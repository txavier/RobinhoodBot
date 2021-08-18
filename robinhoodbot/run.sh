#!/bin/bash

# run bot every hour
while true; do
   # do stuff
   echo $(date)
   python3 main.py
   sleep $[60 * 30]
done
