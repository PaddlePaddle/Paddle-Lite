# on mac
#sed -i "" "s/lite_metal_test::/lite_metal_test::/g" `grep "lite_metal_test::" -rl .`
#sed -i "" "s/namespace lite_metal_test /namespace lite_metal_test /g" `grep "namespace lite_metal_test " -rl .`


# on linux
sed -i  "s/lite/lite_metal/g" lite/model_parser/flatbuffers/param.fbs 
sed -i  "s/lite/lite_metal/g" lite/model_parser/flatbuffers/framework.fbs 

sed -i  "s/lite::/lite_metal::/g" `grep "lite::" -rl ./lite`
sed -i  "s/namespace lite /namespace lite_metal /g" `grep "namespace lite " -rl ./lite`

sed -i "s/ op_type__##/ metal_##op_type__##/g" `grep " op_type__##" -rl ./lite`
sed -i "s/ touch_##/ touch_metal_##/g" `grep " touch_##" -rl ./lite`
sed -i "s/ touch_op_##/ touch_op_metal_##/g" `grep " touch_op_##" -rl ./lite`

sed -i  "s/lite_api::/lite_metal_api::/g" `grep "lite_api::" -rl ./lite`
sed -i  "s/namespace lite_api /namespace lite_metal_api /g" `grep "namespace lite_api " -rl ./lite`

