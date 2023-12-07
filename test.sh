#!/bin/sh

# The URL of the image you want to download
IMAGE_URL="https://batouworkspaceimages.s3.amazonaws.com/3ca68b7f-571a-4472-b33c-5d8f649e1031.png"

# The filename you want to save the image as
OUTPUT_FILE="test.png"

# Use curl to download the image
curl -o $OUTPUT_FILE $IMAGE_URL
