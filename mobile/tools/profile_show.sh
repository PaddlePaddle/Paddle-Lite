#!/usr/bin/env sh
cat <<EOF
<html>
<head>
<style>
html, body {
position: absolute;
width: 100%;
height: 100%;
margin: 0;
}
div.timeview {
width: 100%;
position: relative;
overflow: scroll;
}
ul {
position: absolute;
margin: 0;
list-style:none;
padding: 0;
margin: 0;
}
li {
height: 15px;
position: absolute;
background: blue;
}
li:nth-child(odd) {
background: blue;
}
li:nth-child(even) {
background: rebeccapurple;
}
ul.timeline {
z-index: -1;
}
ul.timeline li {
position: relative;
height: 15px;
width: 100%;
}
ul.timeline li:nth-child(odd) {
background: beige;
}
ul.timeline li:nth-child(even) {
background: antiquewhite;
}
</style>
</head>
<body>
<div class="timeview">
<ul>
EOF

min=$(awk 'NR==1{min=$4} NR>1{if($4 < min) min=$4} END{print min}' $1)
max=$(awk 'NR==1{max=$5} NR>1{if($5 > max) max=$5} END{print max}' $1)
sort $1 -k1,1n | awk -v max="$max" -v min="$min" '
BEGIN {
  total = max - min
}
{
    opid = $1
    optype = $2
    tid = $3
    cb = $4
    ce = $5
    cl = $6
    sum += $4 - $3
    print "<li class=\"timeline\"" \
          " data-opid=\"" opid "\"" \
          " data-optype=\"" optype "\"" \
          " data-tid=\"" tid "\"" \
          " data-begin=\"" cb "\"" \
          " data-end=\"" ce "\"" \
          "></li>"
}
'

cat <<EOF
</ul>
</div>
<pre>
EOF

echo "==================[ profile ]==================="
cat $1 | awk '
NR>1{
    optype = $2
    sum += $5 - $4
    count[$2] += $6
}
END {
for (t in count) {
    msg = sprintf("%-16s\t%-10d\t%-.4f", t, count[t], count[t]*100 / sum);
    print msg
}
}' | sort -k2,2nr
cat $1 | awk '
NR>1{
    sum += $5 - $4
}
END {
msg = sprintf("%-16s\t%-10d\t%-.4f", "total", sum, 100);
print msg
}'

cat <<EOF
</pre>
<script>
const min= $min;
const max= $max;
const px_per_nanosecond = 1/1000000;
const scale = px_per_nanosecond;
const li = document.querySelectorAll('li');
const thread = new Set();
for (let i = 0; i < li.length; i++) {
    const prof = li[i].dataset;
    li[i].style.width = (prof.end - prof.begin)*scale + 'px';
    li[i].style.left = (prof.begin - min)*scale + 'px';
    li[i].style.top = prof.tid * 15 + 'px';
    thread.add(prof.tid);
}
const ul = document.createElement('ul');
ul.classList.add('timeline');
ul.style.width = (max - min)*scale + 'px';
thread.forEach(i => {
    const l = document.createElement('li');
    ul.appendChild(l);
});
const timeview = document.querySelector('.timeview');
timeview.appendChild(ul);
timeview.style.height = thread.size * 15 + 'px';

</script>
</body>
</html>
EOF

