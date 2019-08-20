
module.exports.create = function (args, scope, gl) {
  const output = scope[args.outputs.Out[0]]
  return {
    inferShape() {
      output.dim = output.dim.forEach(d => d === -1 ? 1 : d)
    },
    compute() {
      console.log(output)
    }
  }
}