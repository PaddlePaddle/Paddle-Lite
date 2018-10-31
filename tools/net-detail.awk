BEGIN {
print "digraph G {"
}
/op:/ {
    id++
    opname[id] = $NF
}
/input/ {
    type = "input"
    para = $NF
    if (input[id]) {
        input[id] = input[id] "|"
    }
    input[id] = input[id] "<" para ">" para
}
/output/ {
    type = "output"
    para = $NF
    if (output[id]) {
        output[id] = output[id] "|"
    }
    output[id] = output[id] "<" para ">" para
}
/attr/ {
    type = "attr"
    aname = $NF
    if (attr_key[id]) {
        attr_key[id] = attr_key[id] "|"
        attr_value[id] = attr_value[id] "|"
    }
    attr_key[id] = attr_key[id] $NF
}
/argument/ {
    if (type == "attr") {
        split($0, arr, " - ")
        attr_value[id] = attr_value[id] arr[2]
    } else if ((type == "input") || (type == "output")) {
        if (!var2id[$NF]) {
            var_id++
            var[var_id] = $NF
            var2id[$NF] = var_id
        }
        varid = var2id[$NF]
        lid++
        if (type == "input") {
            line[lid] = "var_" varid " -> " "op_" id ":<" para ">"
            if (xout[$NF]) {
                xi++
                xline[xi] = "xop_" xout[$NF] " -> " "xop_" id
            }
        } else if (type == "output") {
            line[lid] = "op_" id ":<" para ">" " -> " "var_" varid
            xout[$NF] = id
        }
    }
}
/var name/ {
    varname = $NF
    vid = var2id[varname]
}
/var tensor desc dim / {
    if (tensor[vid]) tensor[vid] = tensor[vid] " x "
    tensor[vid] = tensor[vid] $NF
}
END {

print "subgraph cluster_G0 {"
for (i = 1; i <= id; i++) {
    print "xop_" i "[label=\"" i ". " opname[i] "\"]"
}
for (i = 1; i <= xi; i++) {
    print xline[i]
}
print "}"

for (i = 1; i <= id; i++) {
print "op_" i "[group=op;shape=record;label=\"{{" input[i] "}|<op>" i ". " opname[i] "|{" output[i] "}}\"]"
}
for (i = 1; i <= var_id; i++) {
print "var_" i "[label=\"" var[i] " [" tensor[i] "]\"]"
}
for (i = 1; i <= lid; i++) {
print line[i]
}
for (i = 1; i <= id; i++) {
print "attr_" i "[shape=record;label=\"{" attr_key[i] "}|{" attr_value[i] "}\"]"
print "attr_" i " -> " "op_" i ":<op>"
}
print "}"
}

