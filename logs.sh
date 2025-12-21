#!/bin/bash
# logs.sh – Live Log für einen Job
# Usage: bash logs.sh [job_id]  (ohne = neuster Job)

JOB_ID=${1:-$(ls -t /workspace/jobs 2>/dev/null | head -1)}
if [ -z "$JOB_ID" ]; then echo "❌ Keine Jobs gefunden"; exit 1; fi

LOG_FILE="/workspace/jobs/$JOB_ID/out.log"
META_FILE="/workspace/jobs/$JOB_ID/job.json"

echo "=== 📡 Live Log für Job: $JOB_ID ==="
echo "Log: $LOG_FILE"
echo "Drücke Strg+C zum Stoppen"
echo

# Funktion: Status & Fortschritt aus JSON holen
show_header() {
  if [ -f "$META_FILE" ]; then
    STATUS=$(python3 -c "import json; print(json.load(open('$META_FILE')).get('status','unknown'))")
    echo "Status: $STATUS"
    
    # Fortschritt aus Log parsen (z.B. "Processing frame 42/80")
    if [ -f "$LOG_FILE" ]; then
      LAST_FRAME=$(grep -o "Processing frame [0-9]\+" "$LOG_FILE" | tail -1 | grep -o "[0-9]\+")
      TOTAL_FRAMES=$(grep -o "Processing frame [0-9]\+/[0-9]\+" "$LOG_FILE" | tail -1 | grep -o "/[0-9]\+" | tr -d '/')
      if [ -n "$LAST_FRAME" ] && [ -n "$TOTAL_FRAMES" ]; then
        PROGRESS=$((LAST_FRAME * 100 / TOTAL_FRAMES))
        echo "Progress: ${PROGRESS}% (${LAST_FRAME}/${TOTAL_FRAMES} frames)"
      fi
    fi
  fi
  echo "----------------------------------------"
}

# Live anzeigen: Status + letzte 20 Log-Zeilen
while true; do
  clear
  show_header
  if [ -f "$LOG_FILE" ]; then
    tail -n 20 "$LOG_FILE"
  else
    echo "⏳ Warte auf Log-Datei..."
  fi
  sleep 2
done