#!/bin/bash

PROCESS_KEYWORD="mindie"

echo "Searching for processes containing: $PROCESS_KEYWORD"

PROCESSES=$(ps -ef | grep "${PROCESS_KEYWORD}" | grep -v grep | awk '{print $2 "\t" $0}')

if [ -z "$PROCESSES" ]; then
    echo "No matching processes found"
    exit 0
fi

echo "Try to stop all services..."

while IFS= read -r line; do
    if [ -n "$line" ]; then
        PID=$(echo "$line" | awk '{print $1}')
        kill -9 "$PID" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "$line"
        else
            echo "$line (failed)"
        fi
    fi
done <<< "$PROCESSES"

echo "--------"
echo "Done"
