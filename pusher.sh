echo $(git status --porcelain  | awk 'match($1, ""){print $2}' )
for file in $(git status --porcelain  | awk 'match($1, ""){print $2}' );
    do
        git add $file 2> /dev/null
        git commit -m "UPDATE($file) at $(date +%m/%d/%y-%H:%M:%S)"
    done
# git push
