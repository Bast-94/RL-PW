echo $(git status --porcelain | git status --porcelain )
for file in $(git ls-files -m);
    do
        git add $file 2> /dev/null
        git commit -m "UPDATE($file)."
    done
git push