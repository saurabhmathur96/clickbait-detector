for filename in *.txt; do
	echo $filename, `sort $filename | uniq | wc -l`
done
