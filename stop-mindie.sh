#!/bin/bash

PROCESS_KEYWORD="mindieservice_daemon"

echo "Searching for processes containing: $PROCESS_KEYWORD"

PROCESSES=$(ps -ef | grep "${PROCESS_KEYWORD}" | grep -v grep | awk '{print $2 "\t" $0}')

if [ -z "$PROCESSES" ]; then
    echo "No matching processes found"
    exit 0
fi

echo -e "\nFound processes:"
echo -e "PID\tInfo"
echo "------------------------"
echo "${PROCESSES}"

echo -e "\nTerminating processes..."
echo -e "Terminated processes:"
echo -e "PID\tInfo"
echo "------------------------"

while IFS= read -r line; do
    if [ -n "$line" ]; then
        PID=$(echo "$line" | awk '{print $1}')
        kill "$PID" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "$line"
        else
            kill -9 "$PID" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "$line (forced)"
            else
                echo "$line (failed)"
            fi
        fi
    fi
done <<< "$PROCESSES"

echo -e "\nDone"
