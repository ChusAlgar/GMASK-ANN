#!/bin/sh


# Si no recibe argumentos:
if [ $# -eq 0 ]
then
	echo "Es necesario incluir un directorio cuyos archivos .ini ejecutar"


# Si recibe 1 argumento:
elif [ $# -eq 1 ]
then
	archivos_config=$(ls $1)

	for archivo in $archivos_config
	do
		echo $archivo
		$(python3 -m experiments.test_launcher_py3 $archivo)
		$(python2 -m experiments.test_launcher_py2 $archivo)
	done
fi
