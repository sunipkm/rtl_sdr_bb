#!/bin/bash
trap "exit" INT
if ! [[ ( $# == 1 ) ]];
    then printf "Invocation:\n$0 <Directory>\n\n"
    exit
fi
if ! [[ -d $1 ]];
    then echo "$1 is not a directory"
    exit
fi
find $1 -type f -name "*.bzip2" | while read line; do
	dir="$(dirname $line)"
	filename=$(basename -- "$line") # retrieve filename without path
	basename="${filename%.*}" # retrieve base filename, also extracted name
	7za e $line -o$dir # extract bzip2
	mv "$dir/$basename" "$dir/$basename.bin" # rename binary to add extension
	./waterfall.py "$dir/$basename.bin"
	rm -vf "$dir/$basename.bin"
done
