## How can we update flatbuffer module in Paddle-Lite?
Step1: update fbs file
lite/model_parser/flatbuffers/framework.fbs
lite/model_parser/flatbuffers/param.fbs
Step2. update flatbuffer pre-build results
third-party/flatbuffer/update_fbs.sh
Step2. push your update to the remote github repo
git add -f third-party/flatbuffers/pre-build
git commit -m"your commit message"
git push
