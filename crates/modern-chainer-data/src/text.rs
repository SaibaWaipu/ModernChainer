//! Tiny character/word tokenizers and text loaders.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use modern_chainer_core::{Error, Result};

pub const PAD: u32 = 0;
pub const UNK: u32 = 1;
pub const BOS: u32 = 2;
pub const EOS: u32 = 3;

#[derive(Clone, Debug)]
pub struct CharTokenizer {
    chars: Vec<char>,
    stoi: HashMap<char, u32>,
}

impl CharTokenizer {
    pub fn new(corpus: &str) -> Self {
        let chars: Vec<char> = corpus
            .chars()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let stoi = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32 + 4))
            .collect();
        Self { chars, stoi }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len() + 4
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| self.stoi.get(&c).copied().unwrap_or(UNK))
            .collect()
    }

    pub fn encode_with_specials(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![BOS];
        ids.extend(self.encode(text));
        ids.push(EOS);
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| {
                if id < 4 {
                    None
                } else {
                    self.chars.get((id - 4) as usize).copied()
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct WordTokenizer {
    words: Vec<String>,
    stoi: HashMap<String, u32>,
}

impl WordTokenizer {
    pub fn new(corpus: &str) -> Self {
        let words: Vec<String> = corpus
            .split_whitespace()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let stoi = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32 + 4))
            .collect();
        Self { words, stoi }
    }

    pub fn vocab_size(&self) -> usize {
        self.words.len() + 4
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| self.stoi.get(word).copied().unwrap_or(UNK))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| {
                if id < 4 {
                    None
                } else {
                    self.words.get((id - 4) as usize).cloned()
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub fn load_text_lines(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let text = fs::read_to_string(path).map_err(|e| Error::invalid(e.to_string()))?;
    Ok(text.lines().map(str::to_owned).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_roundtrip() {
        let tokenizer = CharTokenizer::new("hello world");
        let ids = tokenizer.encode("hello");
        assert_eq!(tokenizer.decode(&ids), "hello");
        assert_eq!(tokenizer.encode_with_specials("hi")[0], BOS);
    }

    #[test]
    fn word_roundtrip() {
        let tokenizer = WordTokenizer::new("the cat sat on the mat");
        let ids = tokenizer.encode("the cat sat");
        assert_eq!(tokenizer.decode(&ids), "the cat sat");
        assert_eq!(tokenizer.encode("dog"), vec![UNK]);
    }
}
