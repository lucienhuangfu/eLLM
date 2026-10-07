//! `AutoVideoProcessor`：对齐 HuggingFace `video_processing_auto.AutoVideoProcessor`
//! 的功能实现，聚焦 qwen3_5 family 对应的 Qwen3VL 风格视频预处理。
//!
//! Python 侧 `AutoVideoProcessor.from_pretrained` 按 `model_type` 经
//! `VIDEO_PROCESSOR_MAPPING` 分发（qwen3_5 → `Qwen3VLVideoProcessor`）；真正的抽帧 +
//! resize + normalize + patchify 数学在 `transformers.models.qwen3_vl.video_processing_qwen3_vl`。
//! 本模块把该数学移植为纯 Rust，并复用 [`super::image_processor`] 的 `smart_resize`、
//! 双三次重采样、rescale/normalize 与 patchify（图像/视频共享同一 patch 布局契约）。
//!
//! 与图像的差异：
//! - **抽帧**：按 `num_frames` 或 `fps`（配合原生 fps/时长）从已解码帧序列里等距采样，
//!   并把帧数补齐/对齐到 `temporal_patch_size` 的整数倍（`grid_t = nframes / T`）。
//! - **像素预算**：整段视频共享 `max_pixels` 预算，按时间组数摊到每帧，再对每帧
//!   `smart_resize`（近似 HF 的 `max_pixels // nframes * temporal_patch_size`）。
//!
//! 输入为**已解码**的帧序列（HF 的视频处理器同样接收 numpy 帧数组，磁盘解码/抽帧在
//! 上游 decord/qwen-vl-utils 完成），故不引入任何视频解码依赖。

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::image_processor::{
    patchify_frames, rescale_normalize_chw, resize_bicubic_chw, rgb_to_chw_f32, smart_resize,
    RgbImage, SizeDict,
};

/// HF `VIDEO_PROCESSOR_NAME`。
const VIDEO_PROCESSOR_NAME: &str = "video_preprocessor_config.json";
/// HF `PROCESSOR_NAME`（嵌套 processor 配置）。
const PROCESSOR_NAME: &str = "processor_config.json";

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
    vec![0.48145466, 0.4578275, 0.40821073]
}
fn default_image_std() -> Vec<f32> {
    vec![0.26862954, 0.26130258, 0.27577711]
}
fn default_min_pixels() -> usize {
    128 * 28 * 28
}
fn default_max_pixels() -> usize {
    768 * 28 * 28
}
fn default_fps() -> f64 {
    2.0
}

/// 视频处理器配置，对应 `video_preprocessor_config.json`（或嵌套 `processor_config.json`
/// 的 `video_processor` 子对象）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoProcessorConfig {
    #[serde(default)]
    pub video_processor_type: Option<String>,
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
    /// 目标采样帧率；`do_sample_frames` 为真时据此从原生帧率降采样。
    #[serde(default = "default_fps")]
    pub fps: f64,
    /// 显式目标帧数（优先于 fps 采样）。
    #[serde(default)]
    pub num_frames: Option<usize>,
    #[serde(default = "default_true")]
    pub do_sample_frames: bool,
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

impl Default for VideoProcessorConfig {
    fn default() -> Self {
        Self {
            video_processor_type: None,
            patch_size: default_patch_size(),
            merge_size: default_merge_size(),
            temporal_patch_size: default_temporal_patch_size(),
            image_mean: default_image_mean(),
            image_std: default_image_std(),
            min_pixels: None,
            max_pixels: None,
            size: None,
            fps: default_fps(),
            num_frames: None,
            do_sample_frames: true,
            do_resize: true,
            do_rescale: true,
            rescale_factor: default_rescale_factor(),
            do_normalize: true,
            extra: HashMap::new(),
        }
    }
}

impl VideoProcessorConfig {
    /// 由视觉编码器超参构造（patch/merge/temporal 来自 `qwen3_5::VisionConfig`）。
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

    pub fn effective_min_pixels(&self) -> usize {
        self.size
            .as_ref()
            .and_then(|s| s.shortest_edge)
            .or(self.min_pixels)
            .unwrap_or_else(default_min_pixels)
    }

    pub fn effective_max_pixels(&self) -> usize {
        self.size
            .as_ref()
            .and_then(|s| s.longest_edge)
            .or(self.max_pixels)
            .unwrap_or_else(default_max_pixels)
    }

    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }
}

/// 已解码视频帧序列及其时间元数据（用于抽帧）。
#[derive(Debug, Clone, Copy)]
pub struct VideoInput<'a, 'b> {
    /// 已解码帧（假定尺寸一致，RGB HWC）。
    pub frames: &'a [RgbImage<'b>],
    /// 原生帧率（fps 采样时用）。
    pub video_fps: Option<f64>,
    /// 视频总帧数（缺省时用 `frames.len()`）。
    pub total_num_frames: Option<usize>,
    /// 时长（秒）；无 `total_num_frames` 时用于估算。
    pub duration: Option<f64>,
}

impl<'a, 'b> VideoInput<'a, 'b> {
    pub fn new(frames: &'a [RgbImage<'b>]) -> Self {
        Self {
            frames,
            video_fps: None,
            total_num_frames: None,
            duration: None,
        }
    }
}

/// 视频预处理输出。
#[derive(Debug, Clone)]
pub struct VideoProcessorOutput {
    /// `[total_patches, patch_dim]` 行主序展平（与图像同一契约）。
    pub pixel_values: Vec<f32>,
    /// `[num_videos, 3]`，每项为 `[grid_t, grid_h, grid_w]`（patch 单位）。
    pub video_grid_thw: Vec<[usize; 3]>,
    /// 每个 patch 的特征维度 = `C * temporal_patch_size * patch_size * patch_size`。
    pub patch_dim: usize,
    /// 实际采样的帧数（= `grid_t * temporal_patch_size`）。
    pub sampled_frames: usize,
}

impl VideoProcessorOutput {
    pub fn total_patches(&self) -> usize {
        if self.patch_dim == 0 {
            0
        } else {
            self.pixel_values.len() / self.patch_dim
        }
    }
}

/// 依据配置与元数据推算目标采样帧数，并对齐到 `temporal_patch_size` 的整数倍。
fn compute_target_frames(
    cfg: &VideoProcessorConfig,
    video: &VideoInput,
    available: usize,
) -> usize {
    let t = cfg.temporal_patch_size.max(1);

    let raw_target = if !cfg.do_sample_frames {
        available
    } else if let Some(nf) = cfg.num_frames {
        nf
    } else {
        let native_fps = video.video_fps.unwrap_or(cfg.fps).max(1e-6);
        let nframes_in_video = video.total_num_frames.unwrap_or_else(|| {
            video
                .duration
                .map(|d| (d * native_fps).round() as usize)
                .unwrap_or(available)
        });
        // HF：round(nframes_in_video / native_fps * target_fps)
        ((nframes_in_video as f64 / native_fps * cfg.fps).round() as usize).max(1)
    };

    // 对齐到 temporal_patch_size 的整数倍；不足一个时间组则补到一个组。
    let aligned = ((raw_target + t - 1) / t) * t;
    aligned.max(t)
}

/// 从 `available` 帧中等距采样 `target` 帧的索引。
/// `target > available` 时取全部并重复末帧补齐；`target <= available` 时按 linspace 取整。
fn sample_indices(available: usize, target: usize) -> Vec<usize> {
    debug_assert!(available > 0);
    if target >= available {
        let mut idx: Vec<usize> = (0..available).collect();
        while idx.len() < target {
            idx.push(available - 1);
        }
        return idx;
    }
    if target == 1 {
        return vec![0];
    }
    let step = (available - 1) as f64 / (target - 1) as f64;
    (0..target)
        .map(|i| ((i as f64 * step).round() as usize).min(available - 1))
        .collect()
}

/// 视频处理器门面。
#[derive(Debug, Clone)]
pub struct AutoVideoProcessor {
    pub config: VideoProcessorConfig,
}

impl AutoVideoProcessor {
    /// 从模型目录加载：优先嵌套 `processor_config.json` 的 `video_processor` 子对象，
    /// 其次 `video_preprocessor_config.json`（对齐 HF `get_video_processor_config` 优先级）。
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();
        let config = load_video_processor_config(dir)?;
        Ok(Self { config })
    }

    pub fn from_config(config: VideoProcessorConfig) -> Self {
        Self { config }
    }

    /// 预处理一段视频（一批等尺寸已解码帧）。
    pub fn preprocess_video(&self, video: &VideoInput) -> anyhow::Result<VideoProcessorOutput> {
        let cfg = &self.config;
        let p = cfg.patch_size;
        let m = cfg.merge_size;
        let t = cfg.temporal_patch_size;
        if p == 0 || m == 0 || t == 0 {
            anyhow::bail!("patch_size/merge_size/temporal_patch_size must be > 0");
        }
        let available = video.frames.len();
        if available == 0 {
            anyhow::bail!("video has no frames");
        }
        let channels = 3usize;
        let patch_dim = channels * t * p * p;

        let first = video.frames[0];
        if first.data.len() != first.width * first.height * channels {
            anyhow::bail!(
                "frame 0 RGB buffer length {} != width*height*3",
                first.data.len()
            );
        }

        // 1) 抽帧
        let target = compute_target_frames(cfg, video, available);
        let indices = sample_indices(available, target);
        let sampled: Vec<&RgbImage> = indices.iter().map(|&i| &video.frames[i]).collect();
        let grid_t = sampled.len() / t;

        // 2) 逐帧 smart_resize（整段共享像素预算，按时间组数摊到每帧）
        let per_frame_max =
            cfg.effective_max_pixels().max(cfg.effective_min_pixels()) / grid_t.max(1) * t;
        let per_frame_max = per_frame_max.max(cfg.effective_min_pixels());
        let (rh, rw) = if cfg.do_resize {
            smart_resize(
                first.height,
                first.width,
                cfg.factor(),
                cfg.effective_min_pixels(),
                per_frame_max,
            )?
        } else {
            (first.height, first.width)
        };
        if rh % p != 0 || rw % p != 0 || (rh / p) % m != 0 || (rw / p) % m != 0 {
            anyhow::bail!("resized {rh}x{rw} not divisible by patch_size*merge_size ({p}*{m})");
        }

        // 3) resize + rescale + normalize 每帧
        let mut frame_bufs: Vec<Vec<f32>> = Vec::with_capacity(sampled.len());
        for img in &sampled {
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
            frame_bufs.push(frame);
        }

        // 4) patchify（跨全部时间组）
        let frames: Vec<&[f32]> = frame_bufs.iter().map(|f| f.as_slice()).collect();
        let pixel_values = patchify_frames(&frames, channels, t, p, m, rh, rw);
        let video_grid_thw = vec![[grid_t, rh / p, rw / p]];

        Ok(VideoProcessorOutput {
            pixel_values,
            video_grid_thw,
            patch_dim,
            sampled_frames: sampled.len(),
        })
    }
}

/// 读取并解析视频处理器配置 JSON。
fn load_video_processor_config(dir: &Path) -> anyhow::Result<VideoProcessorConfig> {
    let processor_path = dir.join(PROCESSOR_NAME);
    if processor_path.exists() {
        let raw = std::fs::read_to_string(&processor_path)
            .map_err(|e| anyhow::anyhow!("unable to read {}: {e}", processor_path.display()))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("unable to parse {}: {e}", processor_path.display()))?;
        if let Some(sub) = value.get("video_processor") {
            return serde_json::from_value(sub.clone()).map_err(|e| {
                anyhow::anyhow!(
                    "unable to deserialize video_processor in {}: {e}",
                    processor_path.display()
                )
            });
        }
    }

    let video_path = dir.join(VIDEO_PROCESSOR_NAME);
    if video_path.exists() {
        let raw = std::fs::read_to_string(&video_path)
            .map_err(|e| anyhow::anyhow!("unable to read {}: {e}", video_path.display()))?;
        return serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("unable to parse {}: {e}", video_path.display()));
    }

    anyhow::bail!(
        "no video processor config found in {} (expected {PROCESSOR_NAME} or {VIDEO_PROCESSOR_NAME})",
        dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: usize, h: usize, v: u8) -> Vec<u8> {
        vec![v; w * h * 3]
    }

    #[test]
    fn sample_indices_evenly_spaced_and_pads() {
        // 等距采样
        let idx = sample_indices(10, 4);
        assert_eq!(idx.len(), 4);
        assert_eq!(idx[0], 0);
        assert_eq!(*idx.last().unwrap(), 9);
        // 目标超过可用帧：取全部并重复末帧补齐
        let idx = sample_indices(3, 6);
        assert_eq!(idx, vec![0, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn target_frames_aligns_to_temporal_patch_size() {
        let cfg = VideoProcessorConfig::from_vision_params(16, 2, 2);
        let frames_data: Vec<Vec<u8>> = (0..7).map(|i| solid_frame(8, 8, i)).collect();
        let imgs: Vec<RgbImage> = frames_data.iter().map(|d| RgbImage::new(8, 8, d)).collect();
        let video = VideoInput::new(&imgs);
        // num_frames 缺省、fps 缺省路径：available=7 → 对齐到 8
        let target = compute_target_frames(&cfg, &video, imgs.len());
        assert_eq!(target % 2, 0);
        assert!(target >= 2);
    }

    #[test]
    fn preprocess_video_yields_contract_shapes() {
        // 4 帧 64x64；temporal_patch_size=2 → grid_t=2
        let (w, h) = (64usize, 64usize);
        let frames_data: Vec<Vec<u8>> = (0..4).map(|i| solid_frame(w, h, 40 * i)).collect();
        let imgs: Vec<RgbImage> = frames_data.iter().map(|d| RgbImage::new(w, h, d)).collect();
        let mut cfg = VideoProcessorConfig::from_vision_params(16, 2, 2);
        cfg.num_frames = Some(4);
        let proc = AutoVideoProcessor::from_config(cfg);

        let video = VideoInput::new(&imgs);
        let out = proc.preprocess_video(&video).unwrap();

        assert_eq!(out.patch_dim, 3 * 2 * 16 * 16);
        assert_eq!(out.sampled_frames, 4);
        let [gt, gh, gw] = out.video_grid_thw[0];
        assert_eq!(gt, 2);
        assert_eq!(out.total_patches(), gt * gh * gw);
        assert_eq!(out.pixel_values.len(), out.total_patches() * out.patch_dim);
    }
}
