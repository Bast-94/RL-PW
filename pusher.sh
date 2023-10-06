echo $(git status --porcelain  | awk 'match($1, ""){print $2}' )
for file in $(git ls-files -m);
    do
        git add $file 2> /dev/null
        git commit -m "UPDATE($file)."
    done
git push