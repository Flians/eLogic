#!/bin/bash

rm -f DATE26-eLogic-sub/figs/*.drawio.pdf

for f in DATE26-eLogic-sub/figs/*.drawio.png
do
	#/home/flynn/.local/bin/drawio --crop -x -f pdf -t -o ./${f%.*}.pdf ./$f	
	/Applications/draw.io.app/Contents/MacOS/draw.io --crop -x -f pdf -t -o ./${f%.*}.pdf ./$f
done
