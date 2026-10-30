# CPU版 BACKWARD::render の検証・計測メモ

## 要点

- bicycle の固定画像 `_DSC8680`（images_8、618×411）、SfM 初期 Gaussian 54,275 個を使用。optimizer 更新なし。
- CPU版とCUDA版の中間勾配は非ゼロ位置・符号が一致し、NaN/Inf なし。相対 L2 差は約 `10^-7`。
- Gaussian×pixel 候補 35,537,664 回のうち、alpha 判定で 16,925,854 回を除外し、17,066,742 回で勾配を計算した。
- 細粒度タイマーの補正後 0 ms は、処理時間が厳密に 0 という意味ではない。生の測定差が 0 tick だった回数は、再実行後の `results/bicycle_cpu_backward_render_zero_ticks.csv` で確認する。

## 再現手順

```bash
cd submodules/diff-gaussian-rasterization
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../..

python scripts/compare_bicycle_sfm_cpu_cuda_backward_render.py
bash scripts/profile_bicycle_cpu_backward_render.sh
bash scripts/sample_bicycle_cpu_backward_render.sh
```

プロファイルスクリプトは初回ウォームアップ1回、直接計測5回、Nsight Systemsのサンプリング20回を使用する。最後のコマンドには `nsys` が必要。

| 内容 | 出力ファイル |
| --- | --- |
| CPU/CUDA の中間勾配比較 | `results/bicycle_sfm_backward_render_comparison.pt` |
| 全体時間・処理回数 | `results/bicycle_cpu_backward_render_profile.csv` |
| 14区間の直接計測 | `results/bicycle_cpu_backward_render_sections_direct.csv` |
| 各実行・各区間の 0 tick 回数 | `results/bicycle_cpu_backward_render_zero_ticks.csv`（再実行後） |
| Nsight の区間別推定 | `results/bicycle_cpu_backward_render_sections.csv` |
| 絞り込み図 | `results/bicycle_cpu_backward_render_filtering.png` |

<details>
<summary>CPU/CUDA の正しさの検証</summary>

forwardを1回だけ実行し、同じ保存バッファから CPU版とCUDA版の backward render を計算する。`BACKWARD::preprocess` に渡す前の `dL_dmean2D`、`dL_dcolors`、`dL_dopacity`、`dL_dconic2D` を直接比較した。許容誤差による PASS/FAIL 判定はせず、両者の値と絶対差を保存する。

| 勾配 | 最大絶対差 | 平均絶対差 | 相対 L2 差 |
| --- | ---: | ---: | ---: |
| `dL_dmean2D` | 6.98e-10 | 6.33e-13 | 1.08e-7 |
| `dL_dcolors` | 1.40e-9 | 1.26e-13 | 1.08e-7 |
| `dL_dopacity` | 3.73e-9 | 9.39e-13 | 7.79e-8 |
| `dL_dconic2D` | 1.953125e-3 | 1.43e-8 | 1.48e-7 |

全項目で非ゼロ位置と符号が一致し、NaN/Inf はなく、絶対差の中央値は 0。`dL_dconic2D` の最大差は CPU=12629.5537109375、CUDA=12629.5556640625 で、相対差は約 1.55e-7。

同じ Gaussian が複数 tile に現れる。CUDA版は `atomicAdd` の到着順、CPU版は bucket 順で加算するため、浮動小数点の丸め差が生じる。

</details>

<details>
<summary>計測方法とタイマーの限界</summary>

拡張は `-O3 -g` でビルドする。GPU→CPU転送、勾配配列初期化、`render_cpu` 全体、CPU→GPU転送は `std::chrono` で測る。`render_cpu` 内部の14区間は ARM64 の `cntvct_el0` を `ScopedSectionTimer` で読み、区間ごとの tick 合計と実行回数を記録する。

| 区間 | 主な処理 |
| --- | --- |
| Tile / pixel preparation | bucket 情報、256 pixel の状態準備 |
| Gaussian preparation | mean、conic、opacity、勾配蓄積変数の準備 |
| Pixel eligibility check | 画像範囲と last contributor の判定 |
| Gaussian exponent | 座標差、power、`power > 0` 判定 |
| Gaussian weight | `exp(power)` |
| Alpha calculation / threshold | alpha 計算、`alpha < 1/255` 判定 |
| Color / foreground / background alpha gradient | チャネルと alpha の勾配 |
| Transmittance update | `T *= 1 - alpha` |
| Mean / conic / opacity gradient | Gaussian パラメータの勾配 |
| Output accumulation | Gaussian ごとの出力配列へ加算 |

空のタイマーを100,000回×20 batch実行し、batchごとの平均tickの中央値を区間内のタイマー経費とする。ウォームアップ1,000回はこの集計に含めない。別途、タイマーあり／なしのループ差から、記録処理まで含む全体負荷も推定する。

```text
補正後の区間時間 = max(0, 生の区間時間 − 区間内のタイマー経費 × 実行回数)
```

保存済み結果では31,250,000 tick/秒（1 tick = 32 ns）。区間内の経費は約27 ns/回、全体負荷は約57 ns/回。補正に使うのは前者。短い区間ではタイマー経費と処理時間を分離できず、補正後0 msになり得る。分岐の計測範囲は判定だけで、カウンター更新や `continue` は含まない。直接計測は実行を遅くするため、通常実行の時間と混同しない。

</details>

## 計測結果の読み方

| 条件 | `render_cpu` | 注意 |
| --- | ---: | --- |
| 14区間タイマー版、保存済みCSVの5回中央値 | 約23.8 s | 多数の細粒度タイマーを含む |
| 旧メモの `std::chrono` 5回中央値 | 1,160.45 ms | 上とは計測条件が異なるため直接比較しない |

旧メモの後者では、全体1,211.08 ms、GPU→CPU 46.30 ms、配列初期化3.12 ms、CPU→GPU 0.794 ms。これらは保存済みの細粒度タイマー版CSVの値ではない。

処理回数（固定画像1枚）:

```text
Gaussian×pixel候補       35,537,664
├─ 画像範囲外              1,449,626
├─ last contributor以降      95,442
└─ 判定通過               33,992,596
   ├─ power > 0による除外           0
   ├─ alpha < 1/255で除外  16,925,854
   └─ 勾配計算            17,066,742
```

bucket 4,853個、Gaussian splat 138,819個。勾配計算へ進む割合は全候補の48.02%、alpha計算まで進んだ候補の50.21%。

Nsight Systemsのサンプル割合からの推定では、Gaussian exponentが427.76 ms（36.86%）、Alpha thresholdが390.81 ms（33.68%）、Gaussian weightが81.78 ms（7.05%）。これは**サンプリングによる帰属の推定**であり、短い判定1回の実測時間ではない。14区間の全結果は `results/bicycle_cpu_backward_render_sections.csv` を参照。

<details>
<summary>Alpha threshold の分岐予測ミスを調べる場合</summary>

`perf` の `branches:u` と `branch-misses:u` を同じ周期で記録し、`perf annotate` で `alpha < 1/255` に対応する機械語分岐を確認する。近似の分岐予測ミス率は `branch-miss samples / branch samples × 100`。`-O3` や PMU の skid によりソース行との対応がずれることがあるため、機械語アドレスで判断する。

旧メモ作成時の環境では `perf` が未導入で、`perf_event_paranoid=2` のため導入と権限設定が必要だった。具体的なコマンドは環境の `perf` 実体と権限に合わせて決める。

</details>
