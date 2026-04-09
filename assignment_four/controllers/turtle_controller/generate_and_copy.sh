#!/bin/bash

# Check if a basename was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <basename>"
    exit 1
fi

BASENAME=$1

# 1. Generate the maze using the python script
echo "Generating maze with basename: $BASENAME..."
python3 generate_maze.py --basename "$BASENAME"

# Check if the python script succeeded
if [ $? -ne 0 ]; then
    echo "Error: generate_maze.py failed."
    exit 1
fi

# 2. Check if the .wbt file was actually created
if [ ! -f "$BASENAME.wbt" ]; then
    echo "Error: Expected $BASENAME.wbt was not created."
    exit 1
fi

# 3. Copy the .wbt file to the worlds directory
# Based on the structure:
# assignments/assignment_four/controllers/turtle_controller/generate_and_copy.sh
# Target: assignments/assignment_four/worlds/
WORLDS_DIR="../../worlds"

echo "Copying $BASENAME.wbt to $WORLDS_DIR..."
cp "$BASENAME.wbt" "$WORLDS_DIR/"

if [ $? -eq 0 ]; then
    echo "Successfully generated and copied $BASENAME.wbt to $WORLDS_DIR."
else
    echo "Error: Failed to copy $BASENAME.wbt to $WORLDS_DIR."
    exit 1
fi
