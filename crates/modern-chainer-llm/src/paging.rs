use modern_chainer_core::{Backend, DType, Error, Ops, Result, Shape, Tensor};

/// Host-side block allocator with reference counts.
///
/// The v0.4 CPU reference copies block storage on append, so refcounts are
/// bookkeeping for the future in-place/COW kernels rather than a correctness
/// requirement. They are still tested and exposed for observability.
#[derive(Clone, Debug)]
pub struct BlockAllocator {
    block_size: usize,
    refcounts: Vec<u32>,
    free: Vec<usize>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize, block_size: usize) -> Result<Self> {
        if num_blocks == 0 || block_size == 0 {
            return Err(Error::invalid(
                "block allocator needs positive num_blocks and block_size",
            ));
        }
        Ok(Self {
            block_size,
            refcounts: vec![0; num_blocks],
            free: (0..num_blocks).rev().collect(),
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.refcounts.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    pub fn refcount(&self, block: usize) -> Option<u32> {
        self.refcounts.get(block).copied()
    }

    pub fn allocate(&mut self) -> Result<usize> {
        let block = self
            .free
            .pop()
            .ok_or_else(|| Error::invalid("out of paged blocks"))?;
        self.refcounts[block] = 1;
        Ok(block)
    }

    pub fn retain(&mut self, block: usize) -> Result<()> {
        let count = self
            .refcounts
            .get_mut(block)
            .ok_or_else(|| Error::invalid(format!("unknown block {block}")))?;
        if *count == 0 {
            return Err(Error::invalid(format!("retain of free block {block}")));
        }
        *count += 1;
        Ok(())
    }

    pub fn release(&mut self, block: usize) -> Result<()> {
        let count = self
            .refcounts
            .get_mut(block)
            .ok_or_else(|| Error::invalid(format!("unknown block {block}")))?;
        if *count == 0 {
            return Err(Error::invalid(format!("release of free block {block}")));
        }
        *count -= 1;
        if *count == 0 {
            self.free.push(block);
        }
        Ok(())
    }
}

/// Explicit paged KV cache for one layer.
///
/// Blocks are immutable tensors in v0.4: `append` produces new block tensors,
/// which gives fork isolation without a mutable allocator.  The block allocator
/// assigns logical-to-physical mappings; `truncate` returns blocks to it.
#[derive(Clone)]
pub struct PagedKvCache<B: Backend> {
    backend: B,
    k_blocks: Tensor<B>,
    v_blocks: Tensor<B>,
    block_table: Tensor<B>,
    block_size: usize,
    len: usize,
    allocator: BlockAllocator,
}

impl<B: Backend> PagedKvCache<B> {
    pub fn new(
        backend: B,
        block_size: usize,
        num_blocks: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let table = vec![-1i64; num_blocks];
        let block_table = Tensor::from_slice(
            backend.clone(),
            &table,
            Shape::new([num_blocks]),
            DType::I64,
        )?;
        let k_blocks = Tensor::zeros(
            backend.clone(),
            Shape::new([num_blocks, block_size, heads, head_dim]),
            DType::F32,
        )?;
        let v_blocks = Tensor::zeros(
            backend.clone(),
            Shape::new([num_blocks, block_size, heads, head_dim]),
            DType::F32,
        )?;
        Ok(Self {
            backend: backend.clone(),
            k_blocks,
            v_blocks,
            block_table,
            block_size,
            len: 0,
            allocator: BlockAllocator::new(num_blocks, block_size)?,
        })
    }

    pub fn from_dense_with_capacity(
        backend: B,
        k: &Tensor<B>,
        v: &Tensor<B>,
        block_size: usize,
        capacity_blocks: usize,
    ) -> Result<Self> {
        if k.rank() != 3 || v.rank() != 3 {
            return Err(Error::invalid(
                "from_dense_with_capacity expects [heads, seq, head_dim]",
            ));
        }
        if k.shape().dims() != v.shape().dims() {
            return Err(Error::invalid("dense K/V shapes differ"));
        }
        let seq = k.shape().dims()[1];
        let heads = k.shape().dims()[0];
        let head_dim = k.shape().dims()[2];
        let needed = seq.div_ceil(block_size);
        if needed > capacity_blocks {
            return Err(Error::invalid(format!(
                "prompt needs {needed} blocks but capacity is {capacity_blocks}"
            )));
        }
        let mut cache = Self::new(backend, block_size, capacity_blocks, heads, head_dim)?;
        if seq == 0 {
            return Ok(cache);
        }
        let mut table = vec![-1i64; capacity_blocks];
        for entry in table.iter_mut().take(needed) {
            let physical = cache.allocator.allocate()?;
            *entry = physical as i64;
        }
        cache.block_table = Tensor::from_slice(
            cache.backend.clone(),
            &table,
            Shape::new([capacity_blocks]),
            DType::I64,
        )?;
        let (k_blocks, v_blocks) = cache.backend.ops().paged_append(
            &cache.k_blocks,
            &cache.v_blocks,
            &cache.block_table,
            0,
            k,
            v,
        )?;
        cache.k_blocks = k_blocks;
        cache.v_blocks = v_blocks;
        cache.len = seq;
        Ok(cache)
    }

    pub fn k_blocks(&self) -> &Tensor<B> {
        &self.k_blocks
    }

    pub fn v_blocks(&self) -> &Tensor<B> {
        &self.v_blocks
    }

    pub fn block_table(&self) -> &Tensor<B> {
        &self.block_table
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.allocator.num_blocks()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allocator(&self) -> &BlockAllocator {
        &self.allocator
    }

    pub fn append(&mut self, k_new: &Tensor<B>, v_new: &Tensor<B>) -> Result<()> {
        if k_new.rank() != 3 || v_new.rank() != 3 {
            return Err(Error::invalid("append expects [heads, n, head_dim]"));
        }
        if k_new.shape().dims() != v_new.shape().dims() {
            return Err(Error::invalid("append K/V shapes differ"));
        }
        if k_new.shape().dims()[0] != self.k_blocks.shape().dims()[2]
            || k_new.shape().dims()[2] != self.k_blocks.shape().dims()[3]
        {
            return Err(Error::invalid("append K/V head dimensions do not match"));
        }
        let n = k_new.shape().dims()[1];
        if n == 0 {
            return Ok(());
        }
        let current_blocks = self.len.div_ceil(self.block_size);
        let new_len = self.len + n;
        let needed_blocks = new_len.div_ceil(self.block_size);
        let mut table = self.read_table()?;
        let mut allocated = Vec::new();
        for entry in table[current_blocks..needed_blocks].iter_mut() {
            match self.allocator.allocate() {
                Ok(physical) => {
                    *entry = physical as i64;
                    allocated.push(physical);
                }
                Err(error) => {
                    for physical in allocated {
                        let _ = self.allocator.release(physical);
                    }
                    return Err(error);
                }
            }
        }
        let new_table = Tensor::from_slice(
            self.backend.clone(),
            &table,
            Shape::new([table.len()]),
            DType::I64,
        )?;
        let (k_blocks, v_blocks) = self.backend.ops().paged_append(
            &self.k_blocks,
            &self.v_blocks,
            &new_table,
            self.len,
            k_new,
            v_new,
        )?;
        self.k_blocks = k_blocks;
        self.v_blocks = v_blocks;
        self.block_table = new_table;
        self.len = new_len;
        Ok(())
    }

    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.len {
            return Err(Error::invalid(format!(
                "cannot truncate from {} to {new_len}",
                self.len
            )));
        }
        let old_blocks = self.len.div_ceil(self.block_size);
        let new_blocks = new_len.div_ceil(self.block_size);
        let mut table = self.read_table()?;
        for entry in table[new_blocks..old_blocks].iter_mut() {
            let physical = *entry;
            if physical >= 0 {
                self.allocator.release(physical as usize)?;
                *entry = -1;
            }
        }
        self.block_table = Tensor::from_slice(
            self.backend.clone(),
            &table,
            Shape::new([table.len()]),
            DType::I64,
        )?;
        self.len = new_len;
        Ok(())
    }

    pub fn fork(&self) -> Self {
        self.clone()
    }

    fn read_table(&self) -> Result<Vec<i64>> {
        let bytes = self.block_table.to_bytes()?;
        let (chunks, _) = bytes.as_chunks::<8>();
        Ok(chunks
            .iter()
            .map(|chunk| i64::from_ne_bytes(*chunk))
            .collect())
    }
}
