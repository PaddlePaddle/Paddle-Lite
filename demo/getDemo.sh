#!/usr/bin/env bash
wget http://mms-graph.bj.bcebos.com/paddle-mobile%2FPaddleMobile_Android.zip
wget http://mms-graph.bj.bcebos.com/paddle-mobile%2FPaddleMobileDemo_iOS.zip
unzip paddle-mobile%2FPaddleMobile_Android.zip
unzip paddle-mobile%2FPaddleMobileDemo_iOS.zip
rm -rf paddle-mobile%2FPaddleMobile_Android.zip
rm -rf paddle-mobile%2FPaddleMobileDemo_iOS.zip
rm -rf __MACOSX