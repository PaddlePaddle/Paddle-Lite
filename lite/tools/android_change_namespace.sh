sed -i "" "s/lite/lite_$1/g" lite/model_parser/flatbuffers/param.fbs
sed -i  "" "s/lite/lite_$1/g" lite/model_parser/flatbuffers/framework.fbs

sed -i  "" "s/lite/lite_$1/g" lite/backends/opencl/utils/cache.fbs
sed -i  "" "s/lite/lite_$1/g" lite/backends/opencl/utils/tune_cache.fbs

sed -i "" "s/lite::/lite_$1::/g" `grep "lite::" -rl ./lite`
sed -i "" "s/namespace lite /namespace lite_$1 /g" `grep "namespace lite " -rl ./lite`

sed -i "" "s/ op_type__##/ $1_##op_type__##/g" `grep " op_type__##" -rl ./lite`
sed -i "" "s/ touch_##/ touch_$1_##/g" `grep " touch_##" -rl ./lite`
sed -i "" "s/ touch_op_##/ touch_op_$1_##/g" `grep " touch_op_##" -rl ./lite`

sed -i "" "s/lite_api::/lite_$1_api::/g" `grep "lite_api::" -rl ./lite`
sed -i "" "s/::lite_api/::lite_$1_api/g" `grep "::lite_api" -rl ./lite`
sed -i "" "s/namespace lite_api /namespace lite_$1_api /g" `grep "namespace lite_api " -rl ./lite`

sed -i "" "s/namespace lite /namespace lite_$1 /g" `grep "namespace lite " -rl ./third-party/flatbuffers/pre-build`
sed -i "" "s/lite::/lite_$1::/g" `grep "lite::" -rl ./third-party/flatbuffers/pre-build`
sed -i "" "s/lite_api::/lite_$1_api::/g" `grep "lite_api::" -rl ./third-party/flatbuffers/pre-build`
sed -i "" "s/namespace lite_api /namespace lite_$1_api /g" `grep "namespace lite_api " -rl ./third-party/flatbuffers/pre-build`
