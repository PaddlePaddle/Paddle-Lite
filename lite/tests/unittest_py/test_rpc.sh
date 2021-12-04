set -x
port=18812
test_file_name=test_mul_op.py
unittest_path=$(pwd)/op
ROOT_PATH=$(pwd)

function rand {
  min=$1
  max=$(($2-$min+1))
  num=$(($RANDOM+1000000000))
  echo $(($num%$max+$min))
}

function main {

  for i in "$@"; do
    case in 
      --port=*)
          port="${i#*=}"
          shift
          ;;
      --test=*)
          test_file_name="${i#*=}"
          shift
          ;;
      *)
          echo "Unsupported input!"
          exit 1
          ;;
    esac
  done
}

#function run_test {
#  cd ${ROOT_PATH}/rpc_service/
#  # restart rpc service
#  sh start_rpc_server.sh
#}

port_id=$(rand 10000 30000)
echo $port_id

cd ${ROOT_PATH}/
nohup python3.9 server.py --rpc_port=$port_id >/dev/null 2>&1 &
cd $unittest_path/op/
python3.8 $test_file_name --rpc_port=$port_id
#main $@
