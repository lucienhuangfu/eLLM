//! `AutoImageProcessor`：对齐 HuggingFace `image_processing_auto.AutoImageProcessor`
//! 的功能实现，聚焦 qwen3_5（视觉-语言）family 对应的 Qwen2VL 风格图像预处理。
//!
//! Python 侧 `AutoImageProcessor.from_pretrained` 只是按 `model_type` 经
//! `IMAGE_PROCESSOR_MAPPING` 动态分发到具体处理器类（qwen3_5 → `Qwen2VLImageProcessor`）；
//! 真正的预处理数学在 `transformers.models.qwen2_vl.image_processing_qwen2_vl`。
//! 本模块把该数学忠实移植为纯 Rust：`smart_resize` → 双三次重采样 → rescale →
//! normalize → patchify，产出模型 `Qwen3_5VisionPatchEmbed` 期望的 `pixel_values`
//! 与 `image_grid_thw`。
//!
//! 输出契约（由 `scripts/qwen3_5/modeling_qwen3_5.py` 反推并维度自洽验证）：
//! - `pixel_values`：`[total_patches, C * temporal_patch_size * patch_size * patch_size]`，
//!   每行是一个按 `[C, T, P_h, P_w]` 行主序展平的 patch（`PatchEmbed.forward` 里
//!   `view(-1, C, T, P, P)` 的逆）。
//! - `image_grid_thw`：`[num_images, 3]`，单位为 **patch**（`grid_h = resized_h / patch_size`）。
//! - patch 行序：`grid_t → (grid_h/merge) → (grid_w/merge) → merge_h → merge_w`
//!   （对应 Qwen2VL 的 `transpose(0,3,6,4,7,2,1,5,8)`）。
//!
//! 输入为**已解码**的 RGB 像素缓冲（与 HF 一致：其处理器接收 PIL/numpy 已解码数组，
//! 磁盘解码在上游 qwen-vl-utils 完成），故本模块不引入任何图像解码依赖。

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// HF `IMAGE_PROCESSOR_NAME`。
const IMAGE_PROCESSOR_NAME: &str = "preprocessor_config.json";
/// HF `PROCESSOR_NAME`（嵌套 processor 配置）。
const PROCESSOR_NAME: &str = "processor_config.json";
/// `smart_resize` 允许的最大长宽比（HF `MAX_RATIO`）。
const MAX_RATIO: f64 = 200.0;

fn default_patch_size() -> usize {
    16
}
fn default_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}
fn default_true() -> bool {
    true
}
fn default_image_mean() -> Vec<f32> {
    // CLIP / Qwen-VL 标准均值。
    vec![0.48145466, 0.4578275, 0.40821073]
}
fn default_image_std() -> Vec<f32> {
    // CLIP / Qwen-VL 标准方差。
    vec![0.26862954, 0.26130258, 0.27577711]
}
fn default_min_pixels() -> usize {
    56 * 56
}
fn default_max_pixels() -> usize {
    28 * 28 * 1280
}

/// `size` 字段：新版 Qwen2VL 配置用 `shortest_edge`/`longest_edge` 表达
/// min_pixels/max_pixels；旧版直接给顶层 `min_pixels`/`max_pixels`。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SizeDict {
    #[serde(default)]
    pub shortest_edge: Option<usize>,
    #[serde(default)]
    pub longest_edge: Option<usize>,
    #[serde(default)]
    pub height: Option<usize>,
    #[serde(default)]
    pub width: Option<usize>,
}

/// 图像处理器配置，对应 `preprocessor_config.json`（或嵌套 `processor_config.json`
/// 的 `image_processor` 子对象）。字段缺省时回退到 Qwen2VL/qwen3_5 默认值。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessorConfig {
    #[serde(default)]
    pub image_processor_type: Option<String>,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_merge_size")]
    pub merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_image_mean")]
    pub image_mean: Vec<f32>,
    #[serde(default = "default_image_std")]
    pub image_std: Vec<f32>,
    #[serde(default)]
    pub min_pixels: Option<usize>,
    #[serde(default)]
    pub max_pixels: Option<usize>,
    #[serde(default)]
    pub size: Option<SizeDict>,
    #[serde(default = "default_true")]
    pub do_resize: bool,
    #[serde(default = "default_true")]
    pub do_rescale: bool,
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
    #[serde(default = "default_true")]
    pub do_normalize: bool,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            image_processor_type: None,
            patch_size: default_patch_size(),
            merge_size: default_merge_size(),
            temporal_patch_size: default_temporal_patch_size(),
            image_mean: default_image_mean(),
            image_std: default_image_std(),
            min_pixels: None,
            max_pixels: None,
            size: None,
            do_resize: true,
            do_rescale: true,
            rescale_factor: default_rescale_factor(),
            do_normalize: true,
            extra: HashMap::new(),
        }
    }
}

impl ImageProcessorConfig {
    /// 由视觉编码器超参构造（patch/merge/temporal 来自 `qwen3_5::VisionConfig`），
    /// 其余用 Qwen2VL 默认。用于没有 `preprocessor_config.json` 时的便捷路径。
    pub fn from_vision_params(
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
    ) -> Self {
        Self {
            patch_size,
            merge_size,
            temporal_patch_size,
            ..Default::default()
        }
    }

    /// 解析生效的 min_pixels：优先 `size.shortest_edge`，其次顶层 `min_pixels`。
    pub fn effective_min_pixels(&self) -> usize {
        self.size
            .as_ref()
            .and_then(|s| s.shortest_edge)
            .or(self.min_pixels)
            .unwrap_or_else(default_min_pixels)
    }

    /// 解析生效的 max_pixels：优先 `size.longest_edge`，其次顶层 `max_pixels`。
    pub fn effective_max_pixels(&self) -> usize {
        self.size
            .as_ref()
            .and_then(|s| s.longest_edge)
            .or(self.max_pixels)
            .unwrap_or_else(default_max_pixels)
    }

    /// `smart_resize` 的对齐因子 = patch_size * merge_size。
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }
}

/// 已解码的 RGB 图像（HWC，每像素 3 字节）。`data.len()` 必须等于 `width*height*3`。
#[derive(Debug, Clone, Copy)]
pub struct RgbImage<'a> {
    pub width: usize,
    pub height: usize,
    pub data: &'a [u8],
}

impl<'a> RgbImage<'a> {
    pub fn new(width: usize, height: usize, data: &'a [u8]) -> Self {
        Self {
            width,
            height,
            data,
        }
    }
}

/// 图像预处理输出。
#[derive(Debug, Clone)]
pub struct ImageProcessorOutput {
    /// `[total_patches, patch_dim]` 行主序展平。
    pub pixel_values: Vec<f32>,
    /// `[num_images, 3]`，每项为 `[grid_t, grid_h, grid_w]`（patch 单位）。
    pub grid_thw: Vec<[usize; 3]>,
    /// 每个 patch 的特征维度 = `C * temporal_patch_size * patch_size * patch_size`。
    pub patch_dim: usize,
}

impl ImageProcessorOutput {
    /// patch 总行数。
    pub fn total_patches(&self) -> usize {
        if self.patch_dim == 0 {
            0
        } else {
            self.pixel_values.len() / self.patch_dim
        }
    }
}

/// Python `round()`（四舍六入五成双，round-half-to-even）。
fn py_round(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    let fi = f as i64;
    if diff > 0.5 {
        fi + 1
    } else if diff < 0.5 {
        fi
    } else if fi % 2 == 0 {
        fi
    } else {
        fi + 1
    }
}

/// HF `smart_resize`：把 (height, width) 对齐到 `factor` 的整数倍，并夹到
/// `[min_pixels, max_pixels]` 像素预算内（保持长宽比）。
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> anyhow::Result<(usize, usize)> {
    if factor == 0 {
        anyhow::bail!("smart_resize factor must be > 0");
    }
    let h = height as f64;
    let w = width as f64;
    if h <= 0.0 || w <= 0.0 {
        anyhow::bail!("smart_resize got non-positive dimension: {height}x{width}");
    }
    let ratio = h.max(w) / h.min(w);
    if ratio > MAX_RATIO {
        anyhow::bail!("absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}");
    }

    let f = factor as f64;
    let mut h_bar = (py_round(h / f) as f64 * f).max(f) as usize;
    let mut w_bar = (py_round(w / f) as f64 * f).max(f) as usize;

    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = (((h / beta / f).floor()) as usize * factor).max(factor);
        w_bar = (((w / beta / f).floor()) as usize * factor).max(factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil()) as usize * factor;
        w_bar = ((w * beta / f).ceil()) as usize * factor;
    }
    Ok((h_bar, w_bar))
}

/// Keys 双三次核（a = -0.75，匹配 torchvision/PIL BICUBIC）。
fn cubic_weight(x: f32) -> f32 {
    const A: f32 = -0.75;
    let ax = x.abs();
    if ax <= 1.0 {
        ((A + 2.0) * ax - (A + 3.0)) * ax * ax + 1.0
    } else if ax < 2.0 {
        ((A * ax - 5.0 * A) * ax + 8.0 * A) * ax - 4.0 * A
    } else {
        0.0
    }
}

/// 给定映射后的源坐标，返回 4 个采样点（索引已边界钳制）及其权重。
fn cubic_coeffs(coord: f32, len: usize) -> [(usize, f32); 4] {
    let x0 = coord.floor() as i64;
    let last = len as i64 - 1;
    let mut out = [(0usize, 0.0f32); 4];
    for k in 0..4 {
        let tap = x0 - 1 + k as i64;
        let w = cubic_weight(coord - tap as f32);
        let idx = tap.clamp(0, last.max(0)) as usize;
        out[k] = (idx, w);
    }
    out
}

/// 对 CHW 布局的 f32 数据做双三次重采样（可分离：先水平后垂直）。
pub(crate) fn resize_bicubic_chw(
    src: &[f32],
    channels: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    if src_h == dst_h && src_w == dst_w {
        return src.to_vec();
    }
    // 水平 pass: (channels, src_h, src_w) -> (channels, src_h, dst_w)
    let x_scale = src_w as f32 / dst_w as f32;
    let mut tmp = vec![0.0f32; channels * src_h * dst_w];
    for ch in 0..channels {
        for y in 0..src_h {
            let row_base = ch * src_h * src_w + y * src_w;
            let out_base = ch * src_h * dst_w + y * dst_w;
            for x in 0..dst_w {
                let coord = (x as f32 + 0.5) * x_scale - 0.5;
                let mut acc = 0.0f32;
                for (idx, w) in cubic_coeffs(coord, src_w) {
                    acc += src[row_base + idx] * w;
                }
                tmp[out_base + x] = acc;
            }
        }
    }
    // 垂直 pass: (channels, src_h, dst_w) -> (channels, dst_h, dst_w)
    let y_scale = src_h as f32 / dst_h as f32;
    let mut out = vec![0.0f32; channels * dst_h * dst_w];
    for ch in 0..channels {
        for x in 0..dst_w {
            for y in 0..dst_h {
                let coord = (y as f32 + 0.5) * y_scale - 0.5;
                let mut acc = 0.0f32;
                for (idx, w) in cubic_coeffs(coord, src_h) {
                    acc += tmp[ch * src_h * dst_w + idx * dst_w + x] * w;
                }
                out[ch * dst_h * dst_w + y * dst_w + x] = acc;
            }
        }
    }
    out
}

/// RGB u8（HWC）→ f32（CHW）。
pub(crate) fn rgb_to_chw_f32(img: &RgbImage) -> Vec<f32> {
    let n = img.width * img.height;
    let mut out = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[c * n + i] = img.data[i * 3 + c] as f32;
        }
    }
    out
}

/// 就地 rescale + normalize（CHW）。
pub(crate) fn rescale_normalize_chw(
    buf: &mut [f32],
    channels: usize,
    per_channel: usize,
    do_rescale: bool,
    rescale_factor: f32,
    do_normalize: bool,
    mean: &[f32],
    std: &[f32],
) {
    for ch in 0..channels {
        let m = if do_normalize { mean[ch] } else { 0.0 };
        let s = if do_normalize { std[ch] } else { 1.0 };
        let base = ch * per_channel;
        for i in 0..per_channel {
            let mut v = buf[base + i];
            if do_rescale {
                v *= rescale_factor;
            }
            if do_normalize {
                v = (v - m) / s;
            }
            buf[base + i] = v;
        }
    }
}

/// patchify：把 `frames`（每帧 CHW，长度 `channels*resized_h*resized_w`）打包为
/// `[grid_t*grid_h*grid_w, C*T*P*P]` 的展平 patch 序列。
///
/// `frames.len()` 必须为 `temporal_patch_size` 的整数倍（`grid_t = len / t`）；
/// `resized_h`/`resized_w` 必须分别为 `patch_size`、`patch_size*merge_size` 的整数倍。
pub(crate) fn patchify_frames(
    frames: &[&[f32]],
    channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
    merge_size: usize,
    resized_h: usize,
    resized_w: usize,
) -> Vec<f32> {
    let t = temporal_patch_size;
    let p = patch_size;
    let m = merge_size;
    let grid_t = frames.len() / t;
    let grid_h = resized_h / p;
    let grid_w = resized_w / p;
    let hb_count = grid_h / m;
    let wb_count = grid_w / m;
    let patch_dim = channels * t * p * p;
    let per_frame = channels * resized_h * resized_w;

    let rows = grid_t * grid_h * grid_w;
    let mut out = vec![0.0f32; rows * patch_dim];

    for r in 0..rows {
        // 行序：grid_t → hb → wb → mh → mw（mw 最快）
        let mw = r % m;
        let mh = (r / m) % m;
        let wb = (r / (m * m)) % wb_count;
        let hb = (r / (m * m * wb_count)) % hb_count;
        let gt = r / (m * m * wb_count * hb_count);

        let y0 = (hb * m + mh) * p;
        let x0 = (wb * m + mw) * p;
        let row_base = r * patch_dim;

        for ci in 0..channels {
            for ti in 0..t {
                let frame = frames[gt * t + ti];
                let frame_base = ci * resized_h * resized_w;
                for ph in 0..p {
                    for pw in 0..p {
                        let v = frame[frame_base + (y0 + ph) * resized_w + (x0 + pw)];
                        let off = ((ci * t + ti) * p + ph) * p + pw;
                        out[row_base + off] = v;
                    }
                }
            }
        }
        debug_assert_eq!(per_frame, channels * resized_h * resized_w);
    }
    out
}

/// 图像处理器门面。
#[derive(Debug, Clone)]
pub struct AutoImageProcessor {
    pub config: ImageProcessorConfig,
}

impl AutoImageProcessor {
    /// 从模型目录加载：优先嵌套 `processor_config.json` 的 `image_processor` 子对象，
    /// 其次 `preprocessor_config.json`（对齐 HF `get_image_processor_config` 优先级）。
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();
        let config = load_image_processor_config(dir)?;
        Ok(Self { config })
    }

    /// 由显式配置构造。
    pub fn from_config(config: ImageProcessorConfig) -> Self {
        Self { config }
    }

    /// 预处理单张图像（内部按 `temporal_patch_size` 复制为等值帧，`grid_t = 1`）。
    pub fn preprocess_image(&self, img: &RgbImage) -> anyhow::Result<ImageProcessorOutput> {
        self.preprocess_images(&[*img])
    }

    /// 预处理一批图像，`pixel_values` 按图像顺序拼接，`grid_thw` 每图一项。
    pub fn preprocess_images(&self, images: &[RgbImage]) -> anyhow::Result<ImageProcessorOutput> {
        let cfg = &self.config;
        let p = cfg.patch_size;
        let m = cfg.merge_size;
        let t = cfg.temporal_patch_size;
        if p == 0 || m == 0 || t == 0 {
            anyhow::bail!("patch_size/merge_size/temporal_patch_size must be > 0");
        }
        let channels = 3usize;
        let patch_dim = channels * t * p * p;

        let mut pixel_values = Vec::new();
        let mut grid_thw = Vec::with_capacity(images.len());

        for img in images {
            if img.data.len() != img.width * img.height * channels {
                anyhow::bail!(
                    "RGB buffer length {} != width*height*3 ({}*{}*3)",
                    img.data.len(),
                    img.width,
                    img.height
                );
            }

            let (rh, rw) = if cfg.do_resize {
                smart_resize(
                    img.height,
                    img.width,
                    cfg.factor(),
                    cfg.effective_min_pixels(),
                    cfg.effective_max_pixels(),
                )?
            } else {
                (img.height, img.width)
            };
            if rh % p != 0 || rw % p != 0 || (rh / p) % m != 0 || (rw / p) % m != 0 {
                anyhow::bail!("resized {rh}x{rw} not divisible by patch_size*merge_size ({p}*{m})");
            }

            let src = rgb_to_chw_f32(img);
            let mut frame = if cfg.do_resize {
                resize_bicubic_chw(&src, channels, img.height, img.width, rh, rw)
            } else {
                src
            };
            rescale_normalize_chw(
                &mut frame,
                channels,
                rh * rw,
                cfg.do_rescale,
                cfg.rescale_factor,
                cfg.do_normalize,
                &cfg.image_mean,
                &cfg.image_std,
            );

            // 单图：复制为 temporal_patch_size 个等值帧，grid_t = 1。
            let frames: Vec<&[f32]> = vec![frame.as_slice(); t];
            let patches = patchify_frames(&frames, channels, t, p, m, rh, rw);
            pixel_values.extend_from_slice(&patches);
            grid_thw.push([1, rh / p, rw / p]);
        }

        Ok(ImageProcessorOutput {
            pixel_values,
            grid_thw,
            patch_dim,
        })
    }
}

/// 读取并解析图像处理器配置 JSON。
fn load_image_processor_config(dir: &Path) -> anyhow::Result<ImageProcessorConfig> {
    // 1) 嵌套 processor_config.json 的 image_processor 子对象优先。
    let processor_path = dir.join(PROCESSOR_NAME);
    if processor_path.exists() {
        let raw = std::fs::read_to_string(&processor_path)
            .map_err(|e| anyhow::anyhow!("unable to read {}: {e}", processor_path.display()))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("unable to parse {}: {e}", processor_path.display()))?;
        if let Some(sub) = value.get("image_processor") {
            return serde_json::from_value(sub.clone()).map_err(|e| {
                anyhow::anyhow!(
                    "unable to deserialize image_processor in {}: {e}",
                    processor_path.display()
                )
            });
        }
    }

    // 2) 回退到独立 preprocessor_config.json。
    let image_path = dir.join(IMAGE_PROCESSOR_NAME);
    if image_path.exists() {
        let raw = std::fs::read_to_string(&image_path)
            .map_err(|e| anyhow::anyhow!("unable to read {}: {e}", image_path.display()))?;
        return serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("unable to parse {}: {e}", image_path.display()));
    }

    anyhow::bail!(
        "no image processor config found in {} (expected {PROCESSOR_NAME} or {IMAGE_PROCESSOR_NAME})",
        dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_aligns_to_factor_and_clamps_budget() {
        // factor = 16*2 = 32
        let (h, w) = smart_resize(100, 200, 32, 56 * 56, 28 * 28 * 1280).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        // 预算内：应接近原尺寸的对齐值
        assert!((h as i64 - 96).abs() <= 32);
        assert!((w as i64 - 192).abs() <= 32);

        // 超预算缩小
        let (h2, w2) = smart_resize(2000, 2000, 32, 56 * 56, 32 * 32 * 16).unwrap();
        assert!(h2 * w2 <= 32 * 32 * 16 + 32 * 32);
        assert_eq!(h2 % 32, 0);

        // 长宽比过大报错
        assert!(smart_resize(10, 100000, 32, 56 * 56, 28 * 28 * 1280).is_err());
    }

    #[test]
    fn patchify_produces_expected_shape_and_row_order() {
        // 2x2 patch、merge=1、单帧复制 t=2、resized 4x4（grid_h=grid_w=2）
        let p = 2usize;
        let m = 1usize;
        let t = 2usize;
        let rh = 4usize;
        let rw = 4usize;
        let c = 3usize;
        // 构造可辨识的 CHW 帧：值 = channel*100 + y*10 + x
        let mut frame = vec![0.0f32; c * rh * rw];
        for ch in 0..c {
            for y in 0..rh {
                for x in 0..rw {
                    frame[ch * rh * rw + y * rw + x] = (ch * 100 + y * 10 + x) as f32;
                }
            }
        }
        let frames: Vec<&[f32]> = vec![frame.as_slice(); t];
        let out = patchify_frames(&frames, c, t, p, m, rh, rw);
        let rows = (rh / p) * (rw / p); // grid_t=1
        assert_eq!(out.len(), rows * c * t * p * p);

        // 第 0 行应对应 (hb=0,wb=0)，即左上 2x2 patch；行内 [C,T,P,P]
        // 元素 (ci=0,ti=0,ph=0,pw=0) = 像素 (0,0) ch0 = 0
        assert_eq!(out[0], 0.0);
        // (ci=0,ti=0,ph=0,pw=1) = 像素 (0,1) ch0 = 1
        assert_eq!(out[1], 1.0);
        // (ci=0,ti=0,ph=1,pw=0) = 像素 (1,0) ch0 = 10
        let off = ((0 * t + 0) * p + 1) * p + 0;
        assert_eq!(out[off], 10.0);
        // (ci=1,ti=0,ph=0,pw=0) = 像素 (0,0) ch1 = 100
        let off = ((1 * t + 0) * p + 0) * p + 0;
        assert_eq!(out[off], 100.0);
    }

    #[test]
    fn preprocess_image_yields_contract_shapes() {
        // 64x64 纯色图，factor=32 → smart_resize 保持 64x64
        let (w, h) = (64usize, 64usize);
        let data = vec![128u8; w * h * 3];
        let img = RgbImage::new(w, h, &data);
        let cfg = ImageProcessorConfig::from_vision_params(16, 2, 2);
        let proc = AutoImageProcessor::from_config(cfg);
        let out = proc.preprocess_image(&img).unwrap();

        assert_eq!(out.patch_dim, 3 * 2 * 16 * 16);
        // grid = [1, rh/16, rw/16]
        assert_eq!(out.grid_thw.len(), 1);
        let [gt, gh, gw] = out.grid_thw[0];
        assert_eq!(gt, 1);
        assert_eq!(out.total_patches(), gt * gh * gw);
        assert_eq!(out.pixel_values.len(), out.total_patches() * out.patch_dim);
    }

    #[test]
    fn bicubic_resize_identity_when_same_size() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = resize_bicubic_chw(&src, 1, 2, 2, 2, 2);
        assert_eq!(out, src);
    }
}
