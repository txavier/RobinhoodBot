#!/bin/bash

# run bot every hour
while true; do
   # do stuff
   python3 main.py
   sleep $[60 * 60]
done
