if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi

sh "./bin/heatLens_examples.sh" python3

sh "./bin/transport_examples.sh" python3
