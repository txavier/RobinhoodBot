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
      # Test actual DNS resolution of the Robinhood API, not just ICMP ping.
      # On WiFi clusters, ICMP to 8.8.8.8 recovers before DNS/HTTPS does,
      # causing false "password expired" exits during network flaps.
      if nslookup api.robinhood.com > /dev/null 2>&1 && \
         curl -sf --max-time 10 -o /dev/null https://api.robinhood.com/; then
         echo "🚨 Robinhood API is reachable — password required. Stopping bot loop."
         echo "   Run 'python3 main.py' manually to re-authenticate."
         exit 75
      else
         echo "📡 Cannot reach api.robinhood.com. This is likely NOT a password expiration."
         echo "   Continuing loop — will retry when connectivity is restored."
      fi
   fi

   sleep $[60 * 7] # 7 minutes
done