mobilenetv1=../../../../third_party/install/mobilenet_v1
if [ ! -f "$mobilenetv1/__model__" ]; then
  old_pwd=$(pwd)
  cd ../../../../third_party/install
  wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
  tar xzvf mobilenet_v1.tar.gz
  cd $old_pwd
else
  echo "use $mobilenetv1"
fi
./mobilenet_full_api $mobilenetv1 224 224
