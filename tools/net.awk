BEGIN {
    print "digraph {"
}
/op:/ {
    id++
    op = $NF
    opname = op "_" id
    print opname "[\"label\"=\"" op " [" id "]" "\"]"
}
/input/ {
    type = "input"
}
/output/ {
    type = "output"
}
/argument/ {
    if (type == "output") {
        output[$NF] = opname
    } else if (type == "input") {
        if (output[$NF]) {
            print output[$NF] " -> " opname
        }
    }
}
END {
    print "}"
}
