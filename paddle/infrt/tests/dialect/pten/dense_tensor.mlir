// RUN: infrtexec %s | FileCheck %s

// CHECK-LABEL: @basic_tensor
func @basic_tensor() {
  %a = "pten_dt.create_allocator.cpu" (): () -> !pten.CPU_allocator
  %b = "pten_dt.create_context.cpu" (): () -> !pten.CPU_context
  %c = "pten_dt.create_dense_tensor.cpu.f32.nchw" (%a) {dims=[1:i64], lod=[1:i64]}: (!pten.CPU_allocator) -> (!infrt.tensor<X86, NCHW, F32>)
  %d = "pten_kernel.fake_pten_kernel" (%b, %c, %c) {transpose_x=false, transpose_y=false} : (!pten.CPU_context, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> (!infrt.tensor<X86, NCHW, F32>)


  %d = "pten_dt.fake_pten_kernel" (%b, %c, %c) {transpose_x=false} : (!pten.CPU_context, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> (!infrt.tensor<X86, NCHW, F32>)
  infrt.return
}
