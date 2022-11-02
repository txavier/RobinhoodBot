#!/bin/bash

# run bot every hour
while true; do
   # do stuff
   echo $(date)
   # https://stackoverflow.com/questions/40652793/how-to-kill-python-script-with-bash-script
   pkill -f main.py
   python3 main.py
   echo $(date)
   sleep $[60 * 30]
done