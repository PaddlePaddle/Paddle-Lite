function rand {
  min=$1
  max=$(($2-$min+1))
  num=$(($RANDOM+1000000000))
  echo $(($num%$max+$min))
}

if [[ ! -f '.port_id' ]];then
  port_id=$(rand 10000 30000)
  echo $port_id > .port_id
else
  port_id=$(cat .port_id)
fi

# get process-id of current rpc server
RPC_PID=$(lsof -i:$port_id | grep Python | awk -F ' ' '{print $2}')
# restart rpc server
if [[ $RPC_PID != "" ]]; then
  kill -9 ${RPC_PID}
fi

if [[ "$#" == "3" ]];then
  nohup python3.9 server.py --server_ip=$3 >/dev/null 2>&1 &
else
  nohup python3.9 server.py >/dev/null 2>&1 &
fi
