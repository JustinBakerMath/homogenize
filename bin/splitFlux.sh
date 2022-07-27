#!/usr/bin/bash
for i in {3..2}
do
	for j in {5..3}
	do
		for k in {9..7}
		do
			mkdir ./out/splitFluxVol$i$j$k/
			./examples/splitFlux.py --robust --constraint --dout ./out/splitFluxVol$i$j$k/ --vol .$i  \
				--lv -0.$j \
				--beta .$k \
				--scale 5 \
				--tk .01
		done
	done
done
