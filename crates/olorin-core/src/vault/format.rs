//! Vault binary format: header and index entry definitions.

pub const VAULT_MAGIC: [u8; 4] = *b"OLRN";
pub const VAULT_VERSION: u16 = 1;
pub const BLOCK_SIZE: usize = 4096;
pub const HEADER_SIZE: usize = 64;
pub const INDEX_ENTRY_SIZE: usize = 288;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct VaultHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub block_count: u32,
    pub index_offset: u64,
    pub key_id: [u8; 16],
    pub nonce_seed: [u8; 12],
    pub reserved: [u8; 18],
}

impl VaultHeader {
    pub fn new(key_id: [u8; 16], nonce_seed: [u8; 12]) -> Self {
        Self {
            magic: VAULT_MAGIC,
            version: VAULT_VERSION,
            block_count: 0,
            index_offset: HEADER_SIZE as u64,
            key_id,
            nonce_seed,
            reserved: [0u8; 18],
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..10].copy_from_slice(&self.block_count.to_le_bytes());
        buf[10..18].copy_from_slice(&self.index_offset.to_le_bytes());
        buf[18..34].copy_from_slice(&self.key_id);
        buf[34..46].copy_from_slice(&self.nonce_seed);
        buf[46..64].copy_from_slice(&self.reserved);
        buf
    }

    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Result<Self, &'static str> {
        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != VAULT_MAGIC {
            return Err("bad magic");
        }
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version != VAULT_VERSION {
            return Err("unsupported version");
        }
        let block_count = u32::from_le_bytes(buf[6..10].try_into().unwrap());
        let index_offset = u64::from_le_bytes(buf[10..18].try_into().unwrap());
        let mut key_id = [0u8; 16];
        key_id.copy_from_slice(&buf[18..34]);
        let mut nonce_seed = [0u8; 12];
        nonce_seed.copy_from_slice(&buf[34..46]);
        let mut reserved = [0u8; 18];
        reserved.copy_from_slice(&buf[46..64]);
        Ok(Self { magic, version, block_count, index_offset, key_id, nonce_seed, reserved })
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct IndexEntry {
    pub offset: u64,
    pub length: u32,
    pub timestamp: u64,
    pub xxhash: u64,
    pub nonce_counter: u32,
    pub histogram: [u8; 256],
}

impl IndexEntry {
    pub fn to_bytes(&self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; INDEX_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..12].copy_from_slice(&self.length.to_le_bytes());
        buf[12..20].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[20..28].copy_from_slice(&self.xxhash.to_le_bytes());
        buf[28..32].copy_from_slice(&self.nonce_counter.to_le_bytes());
        buf[32..288].copy_from_slice(&self.histogram);
        buf
    }

    pub fn from_bytes(buf: &[u8; INDEX_ENTRY_SIZE]) -> Self {
        let offset = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let length = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let timestamp = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let xxhash = u64::from_le_bytes(buf[20..28].try_into().unwrap());
        let nonce_counter = u32::from_le_bytes(buf[28..32].try_into().unwrap());
        let mut histogram = [0u8; 256];
        histogram.copy_from_slice(&buf[32..288]);
        Self { offset, length, timestamp, xxhash, nonce_counter, histogram }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = VaultHeader::new([0xAA; 16], [0xBB; 12]);
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
        let h2 = VaultHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h2.magic, VAULT_MAGIC);
        let version = { h2.version };
        assert_eq!(version, VAULT_VERSION);
        let bc = { h2.block_count };
        assert_eq!(bc, 0);
        assert_eq!(h2.key_id, [0xAA; 16]);
        assert_eq!(h2.nonce_seed, [0xBB; 12]);
    }

    #[test]
    fn index_entry_roundtrip() {
        let mut hist = [0u8; 256];
        hist[b'a' as usize] = 5;
        hist[b'z' as usize] = 200;
        let e = IndexEntry {
            offset: 64,
            length: 1024,
            timestamp: 1700000000,
            xxhash: 0xDEADBEEF_CAFEBABE,
            nonce_counter: 7,
            histogram: hist,
        };
        let bytes = e.to_bytes();
        assert_eq!(bytes.len(), INDEX_ENTRY_SIZE);
        let e2 = IndexEntry::from_bytes(&bytes);
        assert_eq!(e2.offset, 64);
        assert_eq!(e2.length, 1024);
        assert_eq!(e2.timestamp, 1700000000);
        assert_eq!(e2.xxhash, 0xDEADBEEF_CAFEBABE);
        assert_eq!(e2.nonce_counter, 7);
        assert_eq!(e2.histogram[b'a' as usize], 5);
        assert_eq!(e2.histogram[b'z' as usize], 200);
    }
}
