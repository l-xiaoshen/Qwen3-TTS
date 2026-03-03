{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
   name = "cuda-env-shell";

   buildInputs = with pkgs; [
     cudaPackages.cuda_cudart
     cudaPackages.cuda_nvcc  # Added to provide 'nvcc' for compiling kernels
   ];

   shellHook = ''
     # Set CUDA paths for build tools
     # export CUDA_HOME=${pkgs.cudaPackages.cuda_nvcc}
     # export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH

     # The crucial fix: Point to the actual /lib for cudart, not just /lib/stubs
     # export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.cudaPackages.cuda_nvml_dev.stubs}/lib/stubs:$LD_LIBRARY_PATH

      export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH

     export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
   '';
}
