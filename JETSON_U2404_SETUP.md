Taming_3dgs_u2404環境

CUDA 13.2のconda環境を作成する。

Conda 環境から Jetson の CUDA 13.2 を使うための環境変数設定
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/l4t-gpu-libs/nvgpu:/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}

PyTorchを入れる。（NVIDIA公式のJetson用wheelをダウンロードする必要がある。）
python -m pip install --no-cache-dir \
  torchvision==0.27.1+cu132 \
  --index-url https://download.pytorch.org/whl/cu132

!上記で入れたPyTorchのwheelは、Jetson AGX OrinのGPUの実行コード命令形式であるsm_87が明示的には対応していない。
CUDA拡張をビルドし、Jetson AGX Orin(sm_87)向けGPUバイナリを生成し、インストールする。
export TORCH_CUDA_ARCH_LIST=8.7


CUDA 13.2でビルドするとエラーが出力されるコードの修正（2点
@submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu

1.namespace cg = cooperative_groups;
の直後に以下を追加。
struct MaxUint32
{
        __device__ __forceinline__
        uint32_t operator()(uint32_t a, uint32_t b) const
        {
                return a > b ? a : b;
        }
};

2.last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
↓
last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, MaxUint32());


@submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h

#include <cstdint>の追加。

PythonがCUDA・PyTorch関連の強風ライブラリを見つけられるようにする設定。（これ要らないかも。
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:/opt/nvidia/l4t-gpu-libs/nvgpu:/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"


python -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  ./submodules/diff-gaussian-rasterization

python -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  ./submodules/simple-knn

python -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  ./submodules/fused-ssim

python -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  ./submodules/diff-gaussian-rasterization


残りのパッケージ？をinstall
python -m pip install \
  "numpy<2" \
  "plyfile<1.1.5" \
  tqdm \
  websockets \
  pillow

TensorBoardをinstall（1　iteration毎の処理時間を確認するためのツール）

python -m pip install tensorboard

------------------------------------------------------------------------------
sm_87に対応するPyTorchを入れたconda環境

環境作成

conda create \
  --name taming_3dgs_u2404_sm87 \
  python=3.10 \
  pip \
  -y

conda activate taming_3dgs_u2404_sm87

PyTorchとTorchvisionを入れる。

python -m pip install \
  --pre \
  --no-cache-dir \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu132

JetsonでGPUや消費電力など確認する方法
-------------------------------
bash

export TERM=xterm-256color
sudo -E jtop
-------------------------------
※rootで実行する。
