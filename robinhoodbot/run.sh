#!/bin/bash

# run bot every hour
while true; do
   # do stuff
   echo $(date)
   # https://stackoverflow.com/questions/40652793/how-to-kill-python-script-with-bash-script
   pkill -f main.py
   python3 main.py
   EXIT_CODE=$?
   echo $(date)

   # Exit code 75 = Robinhood session expired, password required.
   # Do NOT retry — repeated login attempts without a password may
   # trigger a security lockout on the Robinhood API.
   if [ $EXIT_CODE -eq 75 ]; then
      echo "🚨 Robinhood password required. Stopping bot loop."
      echo "   Run 'python3 main.py' manually to re-authenticate."
      exit 75
   fi

   sleep $[60 * 7] # 7 minutes
done