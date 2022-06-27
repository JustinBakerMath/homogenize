if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi


echo "Running Heat Lens Examples using $1"

echo "* Single Flux"
FILES="./out/singleFlux/*.pdf"
rm -r $FILES
$1 ./examples/singleFlux.py

echo "* Split Flux Non-Robust"
FILES="./out/splitFlux/*.pdf"
rm -r $FILES
$1 ./examples/splitFlux.py \
	--beta=.9 \
	--scale=5 \
	--tk .001

echo "* Split Flux Robust"
FILES="./out/splitFluxRobust/*.pdf"
rm -r $FILES
$1 ./examples/splitFlux.py --robust --dout ./out/splitFluxRobust/ \
	--beta=.9 \
	--scale=5 \
	--tk .001

echo "* Split Flux Volume"
FILES="./out/splitFluxVol/*.pdf"
rm -r $FILES
$1 ./examples/splitFlux.py --robust --constraint --dout ./out/splitFluxVol/ --vol 0.1  \
	--beta=.9 \
	--scale=5 \
	--tk .001

exit 1

$1 ./examples/splitFlux.py --robust --dout ./out/splitFluxRobust/ 
echo "* Asymmetric Flux Non-Robust"
FILES="./out/asymmetricFlux/*.pdf"
rm -r $FILES
$1 ./examples/asymmetricFlux.py

echo "* Asymmetric Flux Robust"
FILES="./out/asymmetricFluxRobust/*.pdf"
rm -r $FILES
$1 ./examples/asymmetricFlux.py --robust --dout ./out/asymmetricFluxRobust/ 

echo "* Asymmetric Flux Volume"
FILES="./out/asymmetricFluxVol/*.pdf"
rm -r $FILES
$1 ./examples/asymmetricFlux.py --robust --dout ./out/asymmetricFluxVol/ --vol 0.1 

