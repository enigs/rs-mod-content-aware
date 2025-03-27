/// Represents a rectangular region for image cropping operations
///
/// This struct defines a rectangular area within an image using integer coordinates,
/// with the origin (0,0) at the top-left corner. Negative coordinates are possible
/// but will be adjusted during actual cropping operations.
///
/// # Fields
/// * `x: i32` - X-coordinate of the top-left corner of the region
/// * `y: i32` - Y-coordinate of the top-left corner of the region
/// * `width: i32` - Width of the region in pixels
/// * `height: i32` - Height of the region in pixels
///
/// # Example
/// ```rust
/// use content_aware::CropRegion;
///
/// // Create a region starting at (100, 50) with dimensions 800x600
/// let crop = CropRegion {
///     x: 100,
///     y: 50,
///     width: 800,
///     height: 600
/// };
/// ```
#[derive(Debug, Clone)]
pub struct CropRegion {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}