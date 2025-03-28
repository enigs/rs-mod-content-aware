# Content-Aware Image Resizing & Cropping Library

A Rust library that intelligently crops and resizes images by preserving the most visually important content.

## Features

- **Content-Aware Thumbnails**: Generate thumbnails that preserve the most important parts of an image
- **Face Detection**: Automatically detects and prioritizes human faces in portrait images
- **Smart Region Selection**: Uses edge detection, color variance, and other visual cues to identify interesting regions
- **Optimized for Performance**: Multiple optimizations for efficient processing, including adaptive downsampling
- **WebP Output Support**: Produces compressed WebP thumbnails for optimal quality and size
- **Customizable Parameters**: Fine-tune the behavior with detailed configuration options
- **Error Handling**: Comprehensive error types for reliable integration

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
content_aware = { git = "https://github.com/enigs/rs-mod-content-aware", branch = "main" }
```

## Usage

### Basic Example

```rust
use content_aware;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load an image file
    let image_data = fs::read("input.jpg")?;

    // Generate a content-aware thumbnail (300x200 pixels)
    let thumbnail = content_aware::generate_thumbnail(&image_data, 300, 200)?;

    // Save the resulting WebP image
    fs::write("thumbnail.webp", thumbnail)?;

    Ok(())
}
```

### With Custom Configuration

```rust
use content_aware::{generate_thumbnail_with_config, Config};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load an image file
    let image_data = fs::read("group_photo.jpg")?;
    
    // Create a custom face detector configuration
    let config = Config::new()
        .set_min_face_size(30)  // Detect smaller faces
        .set_score_thresh(2.0)  // More lenient detection threshold
        .set_pyramid_scale_factor(0.8);
    
    // Generate a thumbnail with custom face detection settings
    let thumbnail = generate_thumbnail_with_config(
        &image_data, 
        500, 
        300, 
        config
    )?;
    
    // Save the resulting WebP image
    fs::write("group_thumbnail.webp", thumbnail)?;
    
    Ok(())
}
```

### Using Face Detector Directly

```rust
use content_aware::{face_detector, face_detector_with_config, Config};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a detector with default settings
    let mut detector = face_detector()?;
    
    // Or with custom configuration
    let config = Config::new()
        .set_min_face_size(60)
        .set_score_thresh(2.5)
        .set_pyramid_scale_factor(0.8);
    
    let mut custom_detector = face_detector_with_config(config)?;
    
    // Use detectors for custom face detection workflows
    // ...
    
    Ok(())
}
```

## How It Works

The library uses a multi-stage approach to intelligently crop images:

1. **Analysis Phase**: The image is analyzed to identify important visual elements
    - For portraits and people photos, face detection is used to identify subjects
    - For other images, a combination of edge detection, color variation, and center bias is used

2. **Region Selection**: The optimal crop region is determined based on:
    - Detected faces (if present)
    - Visual interest scores across the image
    - Maintaining the target aspect ratio

3. **Thumbnail Creation**: The selected region is cropped and resized to the target dimensions

## API Reference

### Main Functions

- `generate_thumbnail`: Creates a content-aware thumbnail from image bytes using default settings
- `generate_thumbnail_with_config`: Creates a thumbnail with custom face detection configuration
- `face_detector`: Creates a default face detector
- `face_detector_with_config`: Creates a face detector with custom configuration

### Configuration

The `Config` struct allows customizing the face detection algorithm:

```rust
let config = Config::new()
    .set_model_path("path/to/model.bin")  // Default: "./assets/models/seeta_fd_v1.bin"
    .set_min_face_size(40)                // Default: 40
    .set_score_thresh(3.0)                // Default: 3.0
    .set_pyramid_scale_factor(0.7)        // Default: 0.7
    .set_slide_window_step(4, 4);         // Default: (4, 4)
```

### Important Note on Models

This library requires the SeetaFace face detection model file (`seeta_fd_v1.bin`) to be available at the path specified in the configuration. By default, it looks for the model at:

```
./assets/models/seeta_fd_v1.bin
```

You can either:
1. Create this directory structure and place the model file there
2. Customize the path using `Config::set_model_path()`

The SeetaFace model can be downloaded from [their GitHub repository](https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceDetection/model).

## Requirements

- Rust 1.85+
- Required dependencies:
    - image
    - imageproc
    - rustface