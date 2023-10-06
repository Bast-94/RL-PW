for file in $(git ls-files -o);
    do
        git add $file || echo 
        git commit -m "ADD($file)."
    done
for file in $(git ls-files -m);
    do
        git add $file
        git commit -m "UPDATE($file)."
    done
git push