use std::fmt;

/// A namespaced identity for a state slot, e.g. `layers.0.kv.k`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateKey(String);

impl StateKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    pub fn child(&self, child: &str) -> Self {
        if self.0.is_empty() {
            Self(child.to_string())
        } else {
            Self(format!("{}.{}", self.0, child))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for StateKey {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for StateKey {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl AsRef<str> for StateKey {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for StateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A stable, generational handle into a `StateStore`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateId {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

impl StateId {
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    pub fn index(self) -> u32 {
        self.index
    }

    pub fn generation(self) -> u32 {
        self.generation
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "state[{}:{}]", self.index, self.generation)
    }
}
