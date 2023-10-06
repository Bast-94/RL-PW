for file in $(git ls-files -m);
    do
        git add $file
        git commit -m "UPDATE($file)."
    done
git push