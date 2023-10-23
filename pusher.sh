#!/bin/sh
pusher() {
    git add . 2> /dev/null
    echo $(git status --porcelain  | awk 'match($1, ""){print $2}' )
    for file in $(git status --porcelain  | awk 'match($1, ""){print $2}' );
        do
            
            echo "Commiting $file"
            if [ -z "$1" ]; then
                read msg
            fi
            
            git commit -m "UPDATE($file): $msg"
        done
}


pusher $*
