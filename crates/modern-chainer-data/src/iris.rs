//! Iris dataset loader (UCI format).

use std::fs;
use std::path::Path;

use modern_chainer_core::{Error, Result};

#[derive(Clone, Debug)]
pub struct IrisSample {
    pub features: [f32; 4],
    pub label: usize,
}

pub fn load_iris(path: impl AsRef<Path>) -> Result<Vec<IrisSample>> {
    let text = fs::read_to_string(path).map_err(|e| Error::invalid(e.to_string()))?;
    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 {
            return Err(Error::invalid(format!("invalid iris row {}", line_no + 1)));
        }
        let mut features = [0f32; 4];
        for (i, value) in features.iter_mut().enumerate() {
            *value = fields[i].parse().map_err(|_| {
                Error::invalid(format!("invalid iris value at row {}", line_no + 1))
            })?;
        }
        let label = if fields[4].contains("setosa") {
            0
        } else if fields[4].contains("versicolor") {
            1
        } else if fields[4].contains("virginica") {
            2
        } else {
            return Err(Error::invalid(format!("unknown iris class {}", fields[4])));
        };
        out.push(IrisSample { features, label });
    }
    if out.is_empty() {
        return Err(Error::invalid("empty iris dataset"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rows() {
        let dir = std::env::temp_dir().join("mcn-iris-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iris.csv");
        std::fs::write(&path, "5.1,3.5,1.4,0.2,Iris-setosa\n").unwrap();
        let rows = load_iris(path).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].label, 0);
    }
}
