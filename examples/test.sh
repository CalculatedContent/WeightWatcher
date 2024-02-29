#/usr/bin/env bash


notebooks=$(ls *ipynb)
for nb in $notebooks; do
    echo "Processing $nb..."
    nb_out="${nb%.ipynb}.out.ipynb"  # This appends '.out' before '.ipynb'
    papermill "$nb" "$nb_out"
done



      
