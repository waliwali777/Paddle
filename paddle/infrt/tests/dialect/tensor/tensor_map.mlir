func @load_tensor_map() {
  %path = infrt.get_string("/home/chunwei/project/Paddle/cmake-build-debug/multi_fc_model")
  %map = dt.load_params(%path)
  %size = dt.tensor_map_get_size(%map) -> i32
  infrt.print.i32 %size

  %tensor_name = infrt.get_string("x")
  %a = dt.tensor_map_get_tensor(%map, %tensor_name) -> !infrt.tensor<X86, NCHW, F32>

  infrt.return
}
