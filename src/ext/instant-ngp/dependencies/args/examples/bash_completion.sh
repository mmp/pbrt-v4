_args() {
    _init_completion -n 2> /dev/null
    local program comparg

    program="${COMP_WORDS[0]}"
    comparg="--complete" # replace this with your flag

    COMPREPLY=($("$program" "$comparg" bash "$COMP_CWORD" "${COMP_WORDS[@]}" 2> /dev/null))
    [[ $COMPREPLY ]] && return
    _filedir
}

complete -F _args completion
