
/**
 * @file 打包到rd机器的配置
 * @author yangmingming zhangmiao06
 */

const path = require('path');
const ExtractTextPlugin = require('extract-text-webpack-plugin');

const extractLess = new ExtractTextPlugin({
    filename: '[name].css'
});

module.exports = {
    mode: 'development',
    devtool: 'none',
    optimization: {
        minimize: false
    },
    entry: {
        camera: './src/executor/camera',
        index: './src/executor/runner'
    },
    output: {
        filename: '../graphfe/src/view/common/lib/paddle/[name].js',
        path: path.resolve(__dirname, './'),
        library: 'panorama',
        libraryTarget: 'umd',
        libraryExport: 'default'
    },
    module: {
        rules: [{
            test: /\.(eot|woff|woff2|ttf|svg|png|jpg)$/,
            loader: 'url-loader?limit=30000&name=[name].[ext]'
        }, {
            test: /\.less$/,
            exclude: /node_modules/,
            loader: ExtractTextPlugin.extract([
                {loader: 'css-loader', options: {minimize: true}},
                {loader: 'less-loader'}
            ])
        }]
    },
    plugins: [extractLess],
    resolve: {
        extensions: ['.es6', '.js', '.json']
    }
};
