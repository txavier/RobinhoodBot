#!/bin/bash

# run bot every 7 minutes
while true; do
   echo $(date)
   # https://stackoverflow.com/questions/40652793/how-to-kill-python-script-with-bash-script
   pkill -f main.py
   python3 main.py
   EXIT_CODE=$?
   echo $(date)

   # Exit code 75 = Robinhood session expired, password required.
   # Do NOT retry — repeated login attempts without a password may
   # trigger a security lockout on the Robinhood API.
   # But first, verify the network is actually up — if it's down, the
   # login failure is due to connectivity, not an expired session.
   if [ $EXIT_CODE -eq 75 ]; then
      echo "⚠️  Got exit code 75 (password required). Checking network connectivity..."
      if ping -c 1 -W 5 8.8.8.8 > /dev/null 2>&1 || ping -c 1 -W 5 1.1.1.1 > /dev/null 2>&1; then
         echo "🚨 Network is up — Robinhood password required. Stopping bot loop."
         echo "   Run 'python3 main.py' manually to re-authenticate."
         exit 75
      else
         echo "📡 Network appears to be down. This is likely NOT a password expiration."
         echo "   Continuing loop — will retry when connectivity is restored."
      fi
   fi

   sleep $[60 * 7] # 7 minutes
done