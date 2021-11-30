sed -i '' "s/lite/lite_face210/g" lite/model_parser/flatbuffers/param.fbs
sed -i  '' "s/lite/lite_face210/g" lite/model_parser/flatbuffers/framework.fbs

sed -i  '' "s/lite/lite_face210/g" lite/backends/opencl/utils/cache.fbs
sed -i  '' "s/lite/lite_face210/g" lite/backends/opencl/utils/tune_cache.fbs

sed -i '' "s/lite_face210::/lite_face210::/g" `grep "lite_face210::" -rl ./lite`
sed -i '' "s/namespace lite_face210 /namespace lite_face210 /g" `grep "namespace lite_face210 " -rl ./lite`

sed -i '' "s/ face210_##op_type__##/ face210_##op_type__##/g" `grep " face210_##op_type__##" -rl ./lite`
sed -i '' "s/ touch_face210_##/ touch_face210_##/g" `grep " touch_face210_##" -rl ./lite`
sed -i '' "s/ touch_op_face210_##/ touch_op_face210_##/g" `grep " touch_op_face210_##" -rl ./lite`

sed -i '' "s/lite_face210_api::/lite_face210_api::/g" `grep "lite_face210_api::" -rl ./lite`
sed -i '' "s/::lite_face210_api/::lite_face210_api/g" `grep "::lite_face210_api" -rl ./lite`
sed -i '' "s/namespace lite_face210_api /namespace lite_face210_api /g" `grep "namespace lite_face210_api " -rl ./lite`

sed -i '' "s/namespace lite_face210 /namespace lite_face210 /g" `grep "namespace lite_face210 " -rl ./third-party/flatbuffers/pre-build`
sed -i '' "s/lite_face210::/lite_face210::/g" `grep "lite_face210::" -rl ./third-party/flatbuffers/pre-build`
sed -i '' "s/lite_face210_api::/lite_face210_api::/g" `grep "lite_face210_api::" -rl ./third-party/flatbuffers/pre-build`
sed -i '' "s/namespace lite_face210_api /namespace lite_face210_api /g" `grep "namespace lite_face210_api " -rl ./third-party/flatbuffers/pre-build`
