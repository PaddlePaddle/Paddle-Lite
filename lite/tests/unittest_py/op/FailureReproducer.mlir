// configuration: -pass-pipeline='init-return-symbol'
// note: verifyPasses=true
module attributes {".fusion_dump_level" = -1 : si32, ".fusion_dump_tensor" = [], ".shape_mutable" = true, ".toolchain" = "", cn_graph.Arch = "mtp_220.11", cn_graph.target_arch = "x86_64", cn_graph.tfu = true, mm.calibrator = "symmetric", mm.file_path = "./"} {
  func @main(%arg0: tensor<?x?x?x?xi32>) -> tensor<*xi32> {
    %0 = "mm.shape"(%arg0) : (tensor<?x?x?x?xi32>) -> tensor<?xi32>
    %1 = "mm.const"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %2 = "mm.cast"(%1) : (tensor<1xi64>) -> tensor<i32>
    %3 = "mm.fill"(%0, %2) : (tensor<?xi32>, tensor<i32>) -> tensor<*xi32>
    return %3 : tensor<*xi32>
  }
}