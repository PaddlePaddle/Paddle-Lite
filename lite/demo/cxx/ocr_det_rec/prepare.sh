make -j

mkdir ocr_demo
cd ocr_demo

# prepare bin, lib
cp ../ocr_db_crnn .
cp ../run.sh .
cp ../../../../cxx/lib/libpaddle_light_api_shared.so .

# download image and models
wget -c https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/ocr_det_rec_demo/ocr_img_model.tgz
tar zxf ocr_img_model.tgz
rm -rf ocr_img_model.tgz

echo "Prepare ok"
