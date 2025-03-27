/// Configuration options for the face detector
///
/// This struct encapsulates all configurable parameters used to control
/// the behavior and performance characteristics of the face detection algorithm.
#[derive(Clone, Debug)]
pub struct Config {
    /// Path to the face detection model file
    ///
    /// Example: "./assets/models/seeta_fd_v1.bin"
    pub model_path: String,

    /// Minimum size in pixels for face detection
    ///
    /// Faces smaller than this size will not be detected.
    /// Larger values improve performance but might miss smaller faces.
    pub min_face_size: u32,

    /// Confidence threshold for face detection
    ///
    /// Higher values reduce false positives but might increase false negatives.
    /// Range is typically 0-5, with 2-3 being a good balance.
    pub score_thresh: f64,

    /// Scale factor for the detection pyramid
    ///
    /// Controls how aggressively the image is resized between detection scales.
    /// Lower values (closer to 0) are more thorough but slower,
    /// higher values (closer to 1) are faster but might miss faces.
    pub pyramid_scale_factor: f32,

    /// Horizontal step size for the sliding window
    ///
    /// Controls how many pixels the detection window moves horizontally.
    /// Larger values improve performance but might miss some faces.
    pub slide_window_step_x: u32,

    /// Vertical step size for the sliding window
    ///
    /// Controls how many pixels the detection window moves vertically.
    /// Larger values improve performance but might miss some faces.
    pub slide_window_step_y: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: "./assets/models/seeta_fd_v1.bin".to_string(),
            min_face_size: 40,
            score_thresh: 3.0,
            pyramid_scale_factor: 0.7,
            slide_window_step_x: 4,
            slide_window_step_y: 4,
        }
    }
}

impl Config {
    // Create a new instance
    pub fn new() -> Self {
        Self::default()
    }

    // Set model path
    pub fn set_model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = path.into();
        self
    }

    // Set minimum face size
    pub fn set_min_face_size(mut self, size: u32) -> Self {
        self.min_face_size = size;
        self
    }

    // Set score threshold
    pub fn set_score_thresh(mut self, thresh: f64) -> Self {
        self.score_thresh = thresh;
        self
    }

    // Set pyramid scale factor
    pub fn set_pyramid_scale_factor(mut self, factor: f32) -> Self {
        self.pyramid_scale_factor = factor;
        self
    }

    // Set slide window step sizes
    pub fn set_slide_window_step(mut self, x: u32, y: u32) -> Self {
        self.slide_window_step_x = x;
        self.slide_window_step_y = y;
        self
    }

    // Set slide window step x only
    pub fn set_slide_window_step_x(mut self, x: u32) -> Self {
        self.slide_window_step_x = x;
        self
    }

    // Set slide window step y only
    pub fn set_slide_window_step_y(mut self, y: u32) -> Self {
        self.slide_window_step_y = y;
        self
    }
}