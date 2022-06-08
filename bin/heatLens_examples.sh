if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi

echo "Running Heat Lens Examples using $1"

echo "* Single Flux"
$1 ./examples/singleFlux.py

echo "* Split Flux Non-Robust"
$1 ./examples/splitFlux.py

echo "* Split Flux Robust"
$1 ./examples/splitFlux.py --robust --dout ./out/splitFluxRobust/ 

echo "* Asymmetric Flux Non-Robust"
$1 ./examples/asymmetricFlux.py

echo "* Asymmetric Flux Robust"
$1 ./examples/asymmetricFlux.py --robust --dout ./out/asymmetricFluxRobust/ 
