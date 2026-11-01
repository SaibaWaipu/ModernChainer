//! Dataset and iteration primitives.

pub mod audio;
pub mod image;

use modern_chainer_core::Rng;

pub trait Dataset {
    type Item;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<Self::Item>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct VecDataset<T> {
    items: Vec<T>,
}

impl<T> VecDataset<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self { items }
    }

    pub fn into_inner(self) -> Vec<T> {
        self.items
    }
}

impl<T: Clone> Dataset for VecDataset<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<T> {
        self.items.get(index).cloned()
    }
}

/// Epoch-wise batch index iterator with optional shuffling.
pub struct BatchIterator {
    len: usize,
    batch_size: usize,
    shuffle: bool,
    rng: Rng,
    order: Vec<usize>,
    position: usize,
}

impl BatchIterator {
    pub fn new(len: usize, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        assert!(batch_size > 0, "batch_size must be positive");
        let mut order: Vec<usize> = (0..len).collect();
        if shuffle {
            let mut rng = Rng::seed_from_u64(seed);
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            Self {
                len,
                batch_size,
                shuffle,
                rng,
                order,
                position: 0,
            }
        } else {
            Self {
                len,
                batch_size,
                shuffle,
                rng: Rng::seed_from_u64(seed),
                order,
                position: 0,
            }
        }
    }

    pub fn batch_count(&self) -> usize {
        self.len.div_ceil(self.batch_size)
    }

    /// Advance to the next epoch, reshuffling when enabled.
    pub fn next_epoch(&mut self) {
        self.position = 0;
        if self.shuffle {
            for i in (1..self.order.len()).rev() {
                let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
                self.order.swap(i, j);
            }
        }
    }
}

impl Iterator for BatchIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.len {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.len);
        let batch = self.order[self.position..end].to_vec();
        self.position = end;
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_all_indices_exactly_once() {
        let mut sums = [0usize; 10];
        let mut it = BatchIterator::new(10, 3, false, 1);
        let mut total = 0;
        for batch in &mut it {
            total += batch.len();
            for i in batch {
                sums[i] += 1;
            }
        }
        assert_eq!(total, 10);
        assert!(sums.iter().all(|&x| x == 1));
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut seen = [false; 100];
        for batch in BatchIterator::new(100, 7, true, 42) {
            for i in batch {
                assert!(!seen[i]);
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|&x| x));
    }
}
