[中文版](./README_cn.md)
# PaddleJS Tests

Unit and functional tests for Baidu paddle.js can be found in this section.

## Basic Usage

Run  npm run testunits after having run the install, the target operator execution can be specified, and the correctness of the operator execution can be judged according to the test cases of input and calculation output.

```bash
cd web                        # Go to root
npm i                         # Installation dependency
mkdir dist                    # Create resource directory
cd dist                       # Enter resource directory
git clone testunits           # Get test unit data
mv testunits dist             # Move the unit datas to the resource directory
npm run testunits             # run testunits 

```


## Browser coverage

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser