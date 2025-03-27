#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Image(image::ImageError),
    InvalidData(String),
    ModelLoadError(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(err) => write!(f, "IO error: {}", err),
            Error::Image(err) => write!(f, "Image processing error: {}", err),
            Error::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Error::ModelLoadError(msg) => write!(f, "Failed to load model: {}", msg),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            Error::Image(err) => Some(err),
            Error::InvalidData(_) => None,
            Error::ModelLoadError(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<image::ImageError> for Error {
    fn from(err: image::ImageError) -> Self {
        Error::Image(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;