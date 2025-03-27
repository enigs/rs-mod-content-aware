mod config;
mod crop;
mod error;

use image::{GenericImageView, DynamicImage, ImageFormat, imageops::FilterType};
use imageproc::edges;
use rustface::{Detector, FaceInfo, create_detector};
use std::io::Cursor;
use std::cmp;

pub use config::Config;
pub use crop::CropRegion;
pub use error::{Error, Result};

/// Creates a default face detector with standard configuration
///
/// # Returns
/// * `Result<Box<dyn Detector>>` - A configured face detector or an error
///
/// # Example
/// ```rust
/// use content_aware;
///
/// let detector = content_aware::face_detector()?;
/// ```
pub fn face_detector() -> Result<Box<dyn Detector>> {
    face_detector_with_config(Config::new())
}

/// Creates a face detector with custom configuration
///
/// # Parameters
/// * `config: Config` - Custom configuration for the face detector
///
/// # Returns
/// * `Result<Box<dyn Detector>>` - A configured face detector or an error
///
/// # Example
/// ```rust
/// use content_aware::{face_detector_with_config, Config};
///
/// let config = Config::new().set_min_face_size(60);
/// let detector = face_detector_with_config(config)?;
/// ```
pub fn face_detector_with_config(config: Config) -> Result<Box<dyn Detector>> {
    let mut detector = create_detector(&config.model_path)
        .map_err(|e| Error::ModelLoadError(format!("Failed to load model from {}: {}", config.model_path, e)))?;

    detector.set_min_face_size(config.min_face_size);
    detector.set_score_thresh(config.score_thresh);
    detector.set_pyramid_scale_factor(config.pyramid_scale_factor);
    detector.set_slide_window_step(
        config.slide_window_step_x,
        config.slide_window_step_y
    );

    Ok(detector)
}

/// Generate a content-aware thumbnail from image bytes, intelligently cropping and resizing to the target dimensions.
/// The function preserves the most important visual elements in the image using default face detection settings.
///
/// # Parameters
/// * `image_bytes: &[u8]` - The source image data as a byte array
/// * `target_width: u32` - The desired width of the output thumbnail
/// * `target_height: u32` - The desired height of the output thumbnail
///
/// # Returns
/// * `Result<Vec<u8>>` - WebP encoded thumbnail data if successful, or an error
///
/// # Example
/// ```rust
/// use std::fs;
/// use content_aware;
///
/// fn example() -> Result<(), Box<dyn std::error::Error>> {
///     let image_data = fs::read("input.jpg")?;
///     let thumbnail = content_aware::generate_thumbnail(&image_data, 300, 200)?;
///     fs::write("thumbnail.webp", thumbnail)?;
///     Ok(())
/// }
/// ```
pub fn generate_thumbnail(image_bytes: &[u8], target_width: u32, target_height: u32) -> Result<Vec<u8>> {
    // Load image from bytes with minimal processing
    let img = image::load_from_memory(image_bytes)?;

    // Find the best crop region
    let crop_region = find_optimal_crop_region(&img, target_width, target_height)?;

    // Crop and resize efficiently
    let thumbnail = create_thumbnail(&img, crop_region, target_width, target_height)?;

    // Convert to WebP
    let mut buffer = Cursor::new(Vec::new());
    thumbnail.write_to(&mut buffer, ImageFormat::WebP)?;

    let output = buffer.into_inner();
    Ok(output)
}

/// Generate a content-aware thumbnail from image bytes with custom face detection configuration.
/// This allows fine-tuning the face detection behavior to optimize for different image types or performance requirements.
///
/// # Parameters
/// * `image_bytes: &[u8]` - The source image data as a byte array
/// * `target_width: u32` - The desired width of the output thumbnail
/// * `target_height: u32` - The desired height of the output thumbnail
/// * `config: Config` - Custom configuration for the face detector
///
/// # Returns
/// * `Result<Vec<u8>>` - WebP encoded thumbnail data if successful, or an error
///
/// # Example
/// ```rust
/// use std::fs;
/// use content_aware::{generate_thumbnail_with_config, Config};
///
/// fn example() -> Result<(), Box<dyn std::error::Error>> {
///     let image_data = fs::read("group_photo.jpg")?;
///
///     // Create custom configuration for better face detection in group photos
///     let config = Config::new()
///         .set_min_face_size(30)  // Detect smaller faces
///         .set_score_thresh(2.0); // More lenient detection threshold
///
///     let thumbnail = generate_thumbnail_with_config(&image_data, 500, 300, config)?;
///     fs::write("group_thumbnail.webp", thumbnail)?;
///     Ok(())
/// }
/// ```
pub fn generate_thumbnail_with_config(
    image_bytes: &[u8],
    target_width: u32,
    target_height: u32,
    config: Config
) -> Result<Vec<u8>> {
    // Load image from bytes with minimal processing
    let img = image::load_from_memory(image_bytes)?;

    // Find the best crop region using custom face detector configuration
    let crop_region = find_optimal_crop_region_with_config(&img, target_width, target_height, config)?;

    // Crop and resize efficiently
    let thumbnail = create_thumbnail(&img, crop_region, target_width, target_height)?;

    // Convert to WebP
    let mut buffer = Cursor::new(Vec::new());
    thumbnail.write_to(&mut buffer, ImageFormat::WebP)?;

    let output = buffer.into_inner();
    Ok(output)
}

/// Finds the optimal region to crop from the source image based on content analysis
///
/// # Parameters
/// * `img: &DynamicImage` - The source image to analyze
/// * `target_width: u32` - The target width for the crop region
/// * `target_height: u32` - The target height for the crop region
///
/// # Returns
/// * `Result<CropRegion>` - The optimal crop region or an error
///
/// # Example
/// ```rust
/// use image::DynamicImage;
/// use content_aware::find_optimal_crop_region;
///
/// let img = image::open("input.jpg")?;
/// let crop_region = find_optimal_crop_region(&img, 300, 200)?;
/// ```
pub fn find_optimal_crop_region(img: &DynamicImage, target_width: u32, target_height: u32) -> Result<CropRegion> {
    let (img_width, img_height) = img.dimensions();
    let target_ratio = target_width as f32 / target_height as f32;

    // OPTIMIZATION: More aggressive downsampling for analysis
    // Reduced from 1200 to 800 max dimension for faster processing
    const MAX_ANALYSIS_DIMENSION: u32 = 800;

    // OPTIMIZATION: Always downsample for consistent processing time
    let scale_factor = MAX_ANALYSIS_DIMENSION as f32 / cmp::max(img_width, img_height) as f32;
    let new_width = (img_width as f32 * scale_factor).round() as u32;
    let new_height = (img_height as f32 * scale_factor).round() as u32;

    // OPTIMIZATION: Use faster downsampling filter (Triangle instead of Lanczos3)
    let analysis_img = img.resize(new_width, new_height, FilterType::Triangle);

    // Use resized dimensions for calculations
    let (analysis_width, analysis_height) = analysis_img.dimensions();

    // Calculate crop dimensions maintaining aspect ratio
    let (crop_width, crop_height) = if analysis_width as f32 / analysis_height as f32 > target_ratio {
        let crop_height = analysis_height;
        let crop_width = (crop_height as f32 * target_ratio) as u32;
        (crop_width, crop_height)
    } else {
        let crop_width = analysis_width;
        let crop_height = (crop_width as f32 / target_ratio) as u32;
        (crop_width, crop_height)
    };

    // OPTIMIZATION: Quick size-based check to decide analysis method
    // For small images, use simple center crop to save processing time
    if analysis_width <= 200 || analysis_height <= 200 {
        let x = (analysis_width as i32 - crop_width as i32) / 2;
        let y = (analysis_height as i32 - crop_height as i32) / 2;

        // Scale back to original image coordinates
        let scale_factor = img_width as f32 / analysis_width as f32;
        return Ok(CropRegion {
            x: (x as f32 * scale_factor).round() as i32,
            y: (y as f32 * scale_factor).round() as i32,
            width: (crop_width as f32 * scale_factor).round() as i32,
            height: (crop_height as f32 * scale_factor).round() as i32,
        });
    }

    // Try to detect faces first with improved confidence threshold
    // OPTIMIZATION: Use a simpler face detection configuration for speed
    let mut face_detector = face_detector()?;
    let faces = detect_faces_optimized(&analysis_img, &mut face_detector)?;

    // Filter out low-confidence face detections
    let confident_faces: Vec<_> = faces.into_iter()
        .filter(|face| face.score() > 4.0) // Stricter confidence threshold for fewer false positives
        .collect();

    let crop_region = if !confident_faces.is_empty() {
        // Use faces to determine crop region with improved centering
        find_face_centered_crop(analysis_width, analysis_height, &confident_faces, crop_width, crop_height)
    } else {
        // If no faces, find region with the highest entropy/contrast with improved weights
        // OPTIMIZATION: Use faster saliency detection
        find_interesting_region_optimized(&analysis_img, crop_width, crop_height)?
    };

    // Scale back to original image coordinates
    let scale_factor = img_width as f32 / analysis_width as f32;

    Ok(CropRegion {
        x: (crop_region.x as f32 * scale_factor).round() as i32,
        y: (crop_region.y as f32 * scale_factor).round() as i32,
        width: (crop_width as f32 * scale_factor).round() as i32,
        height: (crop_height as f32 * scale_factor).round() as i32,
    })
}

/// Finds the optimal region to crop from the source image based on content analysis with custom face detection configuration
///
/// # Parameters
/// * `img: &DynamicImage` - The source image to analyze
/// * `target_width: u32` - The target width for the crop region
/// * `target_height: u32` - The target height for the crop region
/// * `config: Config` - Custom configuration for the face detector
///
/// # Returns
/// * `Result<CropRegion>` - The optimal crop region or an error
///
/// # Example
/// ```rust
/// use image::DynamicImage;
/// use content_aware::{Config, find_optimal_crop_region_with_config};
///
/// let img = image::open("group_photo.jpg")?;
/// let config = Config::new().set_min_face_size(30).set_score_thresh(2.0);
/// let crop_region = find_optimal_crop_region_with_config(&img, 300, 200, config)?;
/// ```
pub fn find_optimal_crop_region_with_config(
    img: &DynamicImage,
    target_width: u32,
    target_height: u32,
    config: Config
) -> Result<CropRegion> {
    let (img_width, img_height) = img.dimensions();
    let target_ratio = target_width as f32 / target_height as f32;

    // OPTIMIZATION: More aggressive downsampling for analysis
    // Reduced from 1200 to 800 max dimension for faster processing
    const MAX_ANALYSIS_DIMENSION: u32 = 800;

    // OPTIMIZATION: Always downsample for consistent processing time
    let scale_factor = MAX_ANALYSIS_DIMENSION as f32 / cmp::max(img_width, img_height) as f32;
    let new_width = (img_width as f32 * scale_factor).round() as u32;
    let new_height = (img_height as f32 * scale_factor).round() as u32;

    // OPTIMIZATION: Use faster downsampling filter (Triangle instead of Lanczos3)
    let analysis_img = img.resize(new_width, new_height, FilterType::Triangle);

    // Use resized dimensions for calculations
    let (analysis_width, analysis_height) = analysis_img.dimensions();

    // Calculate crop dimensions maintaining aspect ratio
    let (crop_width, crop_height) = if analysis_width as f32 / analysis_height as f32 > target_ratio {
        let crop_height = analysis_height;
        let crop_width = (crop_height as f32 * target_ratio) as u32;
        (crop_width, crop_height)
    } else {
        let crop_width = analysis_width;
        let crop_height = (crop_width as f32 / target_ratio) as u32;
        (crop_width, crop_height)
    };

    // OPTIMIZATION: Quick size-based check to decide analysis method
    // For small images, use simple center crop to save processing time
    if analysis_width <= 200 || analysis_height <= 200 {
        let x = (analysis_width as i32 - crop_width as i32) / 2;
        let y = (analysis_height as i32 - crop_height as i32) / 2;

        // Scale back to original image coordinates
        let scale_factor = img_width as f32 / analysis_width as f32;
        return Ok(CropRegion {
            x: (x as f32 * scale_factor).round() as i32,
            y: (y as f32 * scale_factor).round() as i32,
            width: (crop_width as f32 * scale_factor).round() as i32,
            height: (crop_height as f32 * scale_factor).round() as i32,
        });
    }

    // Try to detect faces using the provided custom configuration
    let mut face_detector = face_detector_with_config(config)?;
    let faces = detect_faces_optimized(&analysis_img, &mut face_detector)?;

    // Filter out low-confidence face detections
    let confident_faces: Vec<_> = faces.into_iter()
        .filter(|face| face.score() > 4.0) // Stricter confidence threshold for fewer false positives
        .collect();

    let crop_region = if !confident_faces.is_empty() {
        // Use faces to determine crop region with improved centering
        find_face_centered_crop(analysis_width, analysis_height, &confident_faces, crop_width, crop_height)
    } else {
        // If no faces, find region with the highest entropy/contrast with improved weights
        // OPTIMIZATION: Use faster saliency detection
        find_interesting_region_optimized(&analysis_img, crop_width, crop_height)?
    };

    // Scale back to original image coordinates
    let scale_factor = img_width as f32 / analysis_width as f32;

    Ok(CropRegion {
        x: (crop_region.x as f32 * scale_factor).round() as i32,
        y: (crop_region.y as f32 * scale_factor).round() as i32,
        width: (crop_width as f32 * scale_factor).round() as i32,
        height: (crop_height as f32 * scale_factor).round() as i32,
    })
}

/// Detects faces in an image with optimizations for processing speed
///
/// # Parameters
/// * `img: &DynamicImage` - The image to analyze for faces
/// * `detector: &mut Box<dyn Detector>` - The face detector to use
///
/// # Returns
/// * `Result<Vec<FaceInfo>>` - A vector of detected faces or an error
///
/// # Example
/// ```rust
/// use image::DynamicImage;
/// use content_aware::{face_detector, detect_faces_optimized};
///
/// let img = image::open("portrait.jpg")?;
/// let mut detector = face_detector()?;
/// let faces = detect_faces_optimized(&img, &mut detector)?;
/// ```
pub fn detect_faces_optimized(img: &DynamicImage, detector: &mut Box<dyn Detector>) -> Result<Vec<FaceInfo>> {
    // OPTIMIZATION: Further downsample large images before face detection
    let (width, height) = img.dimensions();

    let detection_img = if width > 400 || height > 400 {
        let scale = 400.0 / cmp::max(width, height) as f32;
        img.resize(
            (width as f32 * scale) as u32,
            (height as f32 * scale) as u32,
            FilterType::Triangle
        )
    } else {
        img.clone()
    };

    // Convert image to grayscale for faster face detection
    let gray_img = detection_img.to_luma8();

    // Create a buffer that rustface can process
    let (width, height) = gray_img.dimensions();
    let img = gray_img.into_raw();
    let buffer = rustface::ImageData::new(&img, width, height);

    // Detect faces
    Ok(detector.detect(&buffer))
}

/// Finds a crop region centered on detected faces with appropriate padding
///
/// # Parameters
/// * `img_width: u32` - Width of the source image
/// * `img_height: u32` - Height of the source image
/// * `faces: &[FaceInfo]` - Array of detected faces
/// * `crop_width: u32` - Width of the desired crop
/// * `crop_height: u32` - Height of the desired crop
///
/// # Returns
/// * `CropRegion` - The optimal crop region centered on faces
///
/// # Example
/// ```rust
/// use content_aware::{face_detector, detect_faces_optimized, find_face_centered_crop};
/// use image::{DynamicImage, GenericImageView};
///
/// let img = image::open("group_photo.jpg")?;
/// let (width, height) = img.dimensions();
/// let mut detector = face_detector()?;
/// let faces = detect_faces_optimized(&img, &mut detector)?;
/// let crop = find_face_centered_crop(width, height, &faces, 400, 300);
/// ```
pub fn find_face_centered_crop(
    img_width: u32,
    img_height: u32,
    faces: &[FaceInfo],
    crop_width: u32,
    crop_height: u32
) -> CropRegion {
    if faces.is_empty() {
        // Default to center crop if no faces (shouldn't happen due to caller check)
        return CropRegion {
            x: (img_width as i32 - crop_width as i32) / 2,
            y: (img_height as i32 - crop_height as i32) / 2,
            width: crop_width as i32,
            height: crop_height as i32,
        };
    }

    // For a single face, center perfectly on it
    if faces.len() == 1 {
        let face = &faces[0];
        let bbox = face.bbox();
        let face_center_x = bbox.x() + (bbox.width() as i32 / 2);
        let face_center_y = bbox.y() + (bbox.height() as i32 / 2);

        // Calculate vertical offset to place face slightly above center (rule of thirds)
        let vertical_offset = (crop_height as f32 * 0.1) as i32; // 10% above center

        let x = cmp::max(0, face_center_x - (crop_width as i32 / 2));
        let y = cmp::max(0, face_center_y - (crop_height as i32 / 2) - vertical_offset);

        // Adjust if we go out of bounds
        let x = cmp::min(x, img_width as i32 - crop_width as i32);
        let y = cmp::min(y, img_height as i32 - crop_height as i32);

        return CropRegion {
            x,
            y,
            width: crop_width as i32,
            height: crop_height as i32,
        };
    }

    // OPTIMIZATION: For multiple faces, use simpler bounding box approach
    // Rather than weighted calculations
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    // Get the bounding box of all faces
    for face in faces {
        let bbox = face.bbox();
        min_x = cmp::min(min_x, bbox.x());
        min_y = cmp::min(min_y, bbox.y());
        max_x = cmp::max(max_x, bbox.x() + bbox.width() as i32);
        max_y = cmp::max(max_y, bbox.y() + bbox.height() as i32);
    }

    // Add 15% padding
    let width = max_x - min_x;
    let height = max_y - min_y;
    let padding_x = (width as f32 * 0.15) as i32;
    let padding_y = (height as f32 * 0.15) as i32;

    min_x = cmp::max(0, min_x - padding_x);
    min_y = cmp::max(0, min_y - padding_y);
    max_x = cmp::min(img_width as i32, max_x + padding_x);
    max_y = cmp::min(img_height as i32, max_y + padding_y);

    // Calculate center of face group
    let group_center_x = (min_x + max_x) / 2;
    let group_center_y = (min_y + max_y) / 2;

    // Calculate crop position centered on the group
    let x = cmp::max(0, group_center_x - (crop_width as i32 / 2));
    let y = cmp::max(0, group_center_y - (crop_height as i32 / 2));

    // Ensure we don't go out of bounds
    let x = cmp::min(x, img_width as i32 - crop_width as i32);
    let y = cmp::min(y, img_height as i32 - crop_height as i32);

    CropRegion {
        x,
        y,
        width: crop_width as i32,
        height: crop_height as i32,
    }
}

/// Finds regions with interesting visual content when no faces are detected
///
/// # Parameters
/// * `img: &DynamicImage` - The source image to analyze
/// * `crop_width: u32` - The target width for the crop region
/// * `crop_height: u32` - The target height for the crop region
///
/// # Returns
/// * `Result<CropRegion>` - The region with the highest visual interest or an error
///
/// # Example
/// ```rust
/// use image::DynamicImage;
/// use content_aware::find_interesting_region_optimized;
///
/// let img = image::open("landscape.jpg")?;
/// let interesting_region = find_interesting_region_optimized(&img, 800, 600)?;
/// ```
pub fn find_interesting_region_optimized(
    img: &DynamicImage,
    crop_width: u32,
    crop_height: u32
) -> Result<CropRegion> {
    let (img_width, img_height) = img.dimensions();

    // OPTIMIZATION: Even more aggressive downsampling for saliency detection
    let scale_factor = 0.15; // Reduced from 0.25 for faster processing
    let small_img = img.resize(
        (img_width as f32 * scale_factor) as u32,
        (img_height as f32 * scale_factor) as u32,
        FilterType::Nearest, // Faster filter for this stage
    );

    let (small_width, small_height) = small_img.dimensions();
    let small_crop_width = (crop_width as f32 * scale_factor) as u32;
    let small_crop_height = (crop_height as f32 * scale_factor) as u32;

    let mut best_score = f32::NEG_INFINITY;
    let mut best_x = 0;
    let mut best_y = 0;

    // OPTIMIZATION: Use faster edge detection with higher threshold
    let edges = edges::canny(
        &small_img.to_luma8(),
        30.0, // Higher threshold = fewer edges = faster processing
        80.0,
    );

    // Get RGB image for color analysis
    let rgb_img = small_img.to_rgb8();

    // OPTIMIZATION: Check for dominant central object first
    let central_score = check_central_object(&rgb_img, &edges);

    // If there's a strong central object, just return a center crop
    if central_score > 0.6 {
        return Ok(CropRegion {
            x: (img_width as i32 - crop_width as i32) / 2,
            y: (img_height as i32 - crop_height as i32) / 2,
            width: crop_width as i32,
            height: crop_height as i32,
        });
    }

    // OPTIMIZATION: Sparse grid scanning instead of sliding window
    // Define a grid of test points rather than checking every possible position
    let grid_size: u32 = 5; // 5x5 grid = 25 positions to check

    // Handle potential division by zero or small numbers by ensuring minimum values
    let width_range = small_width.saturating_sub(small_crop_width);
    let height_range = small_height.saturating_sub(small_crop_height);

    let steps = grid_size.saturating_sub(1).max(1);
    let x_step = if width_range > 0 { width_range / steps } else { 1 };
    let y_step = if height_range > 0 { height_range / steps } else { 1 };

    // Scan grid positions
    for grid_y in 0..grid_size {
        let y = (grid_y * y_step).min(small_height.saturating_sub(small_crop_height));

        for grid_x in 0..grid_size {
            let x = (grid_x * x_step).min(small_width.saturating_sub(small_crop_width));

            // OPTIMIZATION: Simpler interest calculation
            let score = evaluate_region_interest_fast(
                &edges,
                &rgb_img,
                x,
                y,
                small_crop_width,
                small_crop_height
            );

            if score > best_score {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }

    // Convert back to original image coordinates
    let x = (best_x as f32 / scale_factor) as i32;
    let y = (best_y as f32 / scale_factor) as i32;

    // Ensure the crop region stays within image bounds
    let x = cmp::min(x, img_width as i32 - crop_width as i32);
    let y = cmp::min(y, img_height as i32 - crop_height as i32);

    Ok(CropRegion {
        x,
        y,
        width: crop_width as i32,
        height: crop_height as i32,
    })
}

/// Checks if the image has a dominant central object for simplified cropping
///
/// # Parameters
/// * `rgb_img: &image::RgbImage` - The RGB image to analyze
/// * `edges: &image::GrayImage` - Pre-computed edge detection image
///
/// # Returns
/// * `f32` - A score between 0 and 1 indicating the prominence of a central object
///
/// # Example
/// ```rust
/// use image::{DynamicImage, GenericImageView};
/// use imageproc::edges;
/// use content_aware::check_central_object;
///
/// let img = image::open("object.jpg")?;
/// let rgb_img = img.to_rgb8();
/// let edges = edges::canny(&img.to_luma8(), 30.0, 80.0);
/// let center_score = check_central_object(&rgb_img, &edges);
/// ```
pub fn check_central_object(
    rgb_img: &image::RgbImage,
    edges: &image::GrayImage
) -> f32 {
    let (width, height) = rgb_img.dimensions();

    // Define center region (middle 50%)
    let center_x = width / 4;
    let center_y = height / 4;
    let center_width = width / 2;
    let center_height = height / 2;

    // Count edges in center vs total
    let mut center_edge_count = 0;
    let mut total_edge_count = 0;

    // Sample sparsely (every 2 pixels)
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            if edges.get_pixel(x, y)[0] > 0 {
                total_edge_count += 1;

                // Check if in center region
                if x >= center_x && x < center_x + center_width &&
                    y >= center_y && y < center_y + center_height {
                    center_edge_count += 1;
                }
            }
        }
    }

    if total_edge_count == 0 {
        return 0.5; // No edges - default to middle value
    }

    // Calculate what percentage of edges are in the center
    // Multiply by 4 because center is 1/4 of total area
    let center_ratio = (center_edge_count as f32 / total_edge_count as f32) * 4.0;

    // Return normalized score
    center_ratio.min(1.0)
}

/// Quickly evaluates a region's visual interest using sparse sampling techniques
///
/// # Parameters
/// * `edges: &image::GrayImage` - Pre-computed edge detection image
/// * `rgb_img: &image::RgbImage` - The RGB image to analyze
/// * `x: u32` - X-coordinate of the top-left corner of the region
/// * `y: u32` - Y-coordinate of the top-left corner of the region
/// * `width: u32` - Width of the region to evaluate
/// * `height: u32` - Height of the region to evaluate
///
/// # Returns
/// * `f32` - A score indicating the visual interest of the region
///
/// # Example
/// ```rust
/// use image::{DynamicImage, GenericImageView};
/// use imageproc::edges;
/// use content_aware::evaluate_region_interest_fast;
///
/// let img = image::open("scene.jpg")?;
/// let rgb_img = img.to_rgb8();
/// let edges = edges::canny(&img.to_luma8(), 30.0, 80.0);
/// let score = evaluate_region_interest_fast(&edges, &rgb_img, 100, 100, 400, 300);
/// ```
pub fn evaluate_region_interest_fast(
    edges: &image::GrayImage,
    rgb_img: &image::RgbImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32
) -> f32 {
    // OPTIMIZATION: Sparse sampling
    // Sample only 16 points in a 4x4 grid for superfast analysis

    let step_x = width / 4;
    let step_y = height / 4;

    let mut edge_count = 0;
    let mut color_variance = 0.0;
    let mut saturation_sum = 0.0;

    // Use a simple fixed-size sampling pattern
    for i in 0..4 {
        for j in 0..4 {
            let px = x + j * step_x + step_x / 2;
            let py = y + i * step_y + step_y / 2;

            // Boundary check
            if px >= edges.width() || py >= edges.height() {
                continue;
            }

            // Edge detection
            if edges.get_pixel(px, py)[0] > 0 {
                edge_count += 1;
            }

            // Color variance (very simple approximation)
            let rgb = rgb_img.get_pixel(px, py);
            let r = rgb[0] as f32;
            let g = rgb[1] as f32;
            let b = rgb[2] as f32;

            // Simplified variance approximation
            let avg = (r + g + b) / 3.0;
            let variance = ((r - avg).powi(2) + (g - avg).powi(2) + (b - avg).powi(2)) / 3.0;
            color_variance += variance;

            // Simple saturation calculation
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            if max > 0.0 {
                saturation_sum += (max - min) / max;
            }
        }
    }

    // Normalize scores
    let edge_density = edge_count as f32 / 16.0;
    let color_variance_score = (color_variance / 16.0 / 10000.0).min(1.0);
    let saturation_score = saturation_sum / 16.0;

    // Simple center bias (distance from image center)
    let center_x = x as f32 + width as f32 / 2.0;
    let center_y = y as f32 + height as f32 / 2.0;
    let img_center_x = rgb_img.width() as f32 / 2.0;
    let img_center_y = rgb_img.height() as f32 / 2.0;
    let dx = (center_x - img_center_x) / rgb_img.width() as f32;
    let dy = (center_y - img_center_y) / rgb_img.height() as f32;
    let distance = (dx * dx + dy * dy).sqrt();
    let center_bias = 1.0 - distance.min(1.0);

    // OPTIMIZATION: Simplified weights balanced for speed and quality
    0.25 * edge_density +
        0.25 * saturation_score +
        0.25 * color_variance_score +
        0.25 * center_bias
}

/// Creates a thumbnail by cropping and resizing the image with bounds checking
///
/// # Parameters
/// * `img: &DynamicImage` - The source image
/// * `crop: CropRegion` - The region to crop from the source image
/// * `target_width: u32` - The target width for the final thumbnail
/// * `target_height: u32` - The target height for the final thumbnail
///
/// # Returns
/// * `Result<DynamicImage>` - The cropped and resized thumbnail or an error
///
/// # Example
/// ```rust
/// use image::DynamicImage;
/// use content_aware::{CropRegion, create_thumbnail};
///
/// let img = image::open("input.jpg")?;
/// let crop = CropRegion { x: 100, y: 50, width: 800, height: 600 };
/// let thumbnail = create_thumbnail(&img, crop, 300, 200)?;
/// thumbnail.save("thumbnail.jpg")?;
/// ```
pub fn create_thumbnail(
    img: &DynamicImage,
    crop: CropRegion,
    target_width: u32,
    target_height: u32
) -> Result<DynamicImage> {
    // Ensure crop coordinates are non-negative (required for crop_imm)
    let x = cmp::max(0, crop.x) as u32;
    let y = cmp::max(0, crop.y) as u32;

    // Ensure crop dimensions are within image bounds
    let (img_width, img_height) = img.dimensions();
    let width = cmp::min(crop.width as u32, img_width - x);
    let height = cmp::min(crop.height as u32, img_height - y);

    // OPTIMIZATION: Use faster resize filter for final output
    // Crop the image with bounds checking
    let cropped = img.crop_imm(x, y, width, height);

    // Resize to final dimensions with a good balance of speed and quality
    Ok(cropped.resize_exact(
        target_width,
        target_height,
        FilterType::Triangle,  // Faster than Lanczos3 with minimal quality loss
    ))
}