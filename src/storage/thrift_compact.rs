//! Hand-rolled Thrift compact-protocol decoder. Used by `parquet.rs` for
//! the file-footer FileMetaData struct. Pulling in the `thrift` crate
//! would violate the 2-dependency rule; the spec is short and the subset
//! we need is small.

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CompactType {
    Stop, True, False, I8, I16, I32, I64,
    Double, Binary, List, Set, Map, Struct,
    Other(u8),
}

impl CompactType {
    fn from_byte(b: u8) -> Self {
        match b {
            0  => Self::Stop,   1 => Self::True,   2 => Self::False,
            3  => Self::I8,     4 => Self::I16,    5 => Self::I32,
            6  => Self::I64,    7 => Self::Double, 8 => Self::Binary,
            9  => Self::List,  10 => Self::Set,   11 => Self::Map,
            12 => Self::Struct,
            o  => Self::Other(o),
        }
    }
}

pub(crate) struct ThriftReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ThriftReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }

    fn read_byte(&mut self) -> Result<u8, String> {
        let b = *self.bytes.get(self.pos).ok_or("unexpected end of footer")?;
        self.pos += 1;
        Ok(b)
    }

    fn read_varint_u64(&mut self) -> Result<u64, String> {
        let mut result: u64 = 0;
        let mut shift = 0;
        for _ in 0..10 {
            let b = self.read_byte()?;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 { return Ok(result); }
            shift += 7;
        }
        Err("varint exceeds 10 bytes".into())
    }

    pub(crate) fn read_zigzag_i32(&mut self) -> Result<i32, String> {
        let v = self.read_varint_u64()?;
        Ok(((v >> 1) as i32) ^ -((v & 1) as i32))
    }

    pub(crate) fn read_zigzag_i64(&mut self) -> Result<i64, String> {
        let v = self.read_varint_u64()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }

    fn read_double(&mut self) -> Result<f64, String> {
        if self.pos + 8 > self.bytes.len() { return Err("short double".into()); }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.bytes[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(f64::from_le_bytes(buf))
    }

    pub(crate) fn read_binary(&mut self) -> Result<&'a [u8], String> {
        let len = self.read_varint_u64()? as usize;
        if self.pos + len > self.bytes.len() { return Err("short binary".into()); }
        let s = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(s)
    }

    pub(crate) fn read_list_header(&mut self) -> Result<(CompactType, usize), String> {
        let b = self.read_byte()?;
        let elem_type = CompactType::from_byte(b & 0x0f);
        let size_hi = (b >> 4) & 0x0f;
        let size = if size_hi == 0x0f {
            self.read_varint_u64()? as usize
        } else {
            size_hi as usize
        };
        Ok((elem_type, size))
    }

    pub(crate) fn skip_value(&mut self, ty: CompactType) -> Result<(), String> {
        match ty {
            CompactType::True | CompactType::False | CompactType::Stop => {}
            CompactType::I8                       => { self.read_byte()?; }
            CompactType::I16 | CompactType::I32   => { self.read_zigzag_i32()?; }
            CompactType::I64                      => { self.read_zigzag_i64()?; }
            CompactType::Double                   => { self.read_double()?; }
            CompactType::Binary                   => { self.read_binary()?; }
            CompactType::List | CompactType::Set  => {
                let (et, n) = self.read_list_header()?;
                for _ in 0..n { self.skip_value(et)?; }
            }
            CompactType::Map => {
                let n = self.read_varint_u64()? as usize;
                if n > 0 {
                    let kv = self.read_byte()?;
                    let kt = CompactType::from_byte((kv >> 4) & 0x0f);
                    let vt = CompactType::from_byte(kv & 0x0f);
                    for _ in 0..n {
                        self.skip_value(kt)?;
                        self.skip_value(vt)?;
                    }
                }
            }
            CompactType::Struct => { self.skip_struct()?; }
            CompactType::Other(o) => return Err(format!("unknown compact type {o}")),
        }
        Ok(())
    }

    fn skip_struct(&mut self) -> Result<(), String> {
        loop {
            let h = self.read_byte()?;
            if h == 0 { return Ok(()); }
            let ty = CompactType::from_byte(h & 0x0f);
            if (h >> 4) == 0 { self.read_zigzag_i32()?; }
            self.skip_value(ty)?;
        }
    }

    pub(crate) fn read_field_header(
        &mut self,
        last_field_id: i16,
    ) -> Result<Option<(i16, CompactType)>, String> {
        let h = self.read_byte()?;
        if h == 0 { return Ok(None); }
        let ty = CompactType::from_byte(h & 0x0f);
        let delta = (h >> 4) as i16;
        let field_id = if delta != 0 {
            last_field_id + delta
        } else {
            self.read_zigzag_i32()? as i16
        };
        Ok(Some((field_id, ty)))
    }
}
