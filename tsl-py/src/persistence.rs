//! Saves and loads a fitted `TSL` as a bincode file.

use std::fs;

use pyo3::{
    exceptions::{PyIOError, PyValueError},
    prelude::*,
};
use tsl::forest::TSL;

/// Writes `tsl` to `path` in bincode, which preserves exact floating point values.
pub fn save(tsl: &TSL, path: &str) -> PyResult<()> {
    let bytes = bincode::serialize(tsl)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize TSL model: {}", e)))?;
    fs::write(path, bytes)
        .map_err(|e| PyIOError::new_err(format!("Failed to write to file {}: {}", path, e)))?;
    Ok(())
}

/// Reads a `TSL` written by [`save`] from `path`.
pub fn load(path: &str) -> PyResult<TSL> {
    let bytes = fs::read(path)
        .map_err(|e| PyIOError::new_err(format!("Failed to read from file {}: {}", path, e)))?;
    let tsl: TSL = bincode::deserialize(&bytes)
        .map_err(|e| PyValueError::new_err(format!("Failed to deserialize TSL model: {}", e)))?;
    Ok(tsl)
}
