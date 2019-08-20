module.exports.create = function (args, scope, gl) {
  const output = scope[args.outputs.Out[0]]
  const input = scope[args.inputs.X[0]]
  return {
    inferShape() {
      output.dim = output.dim.map(d => d === -1 ? 1 : d)
    },
    compute() {
      console.log(input)
      console.log(output)
    }
  }
}


