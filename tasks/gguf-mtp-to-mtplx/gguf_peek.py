#!/usr/bin/env python3
"""Minimal GGUF header reader: lists tensor name / dtype / shape / offset.
Reads only the header + tensor-info section (no tensor data), so a partial
download of the file head is enough. Pure stdlib."""
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# value type enum (gguf)
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING,
 ARRAY, UINT64, INT64, FLOAT64) = range(13)

# ggml tensor dtype enum (subset we care about)
GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 30: "BF16",
}


class R:
    def __init__(self, b):
        self.b = b
        self.o = 0

    def need(self, n):
        if self.o + n > len(self.b):
            raise EOFError(f"need {n} bytes at {self.o}, have {len(self.b)-self.o}")

    def u8(self):  self.need(1); v = self.b[self.o]; self.o += 1; return v
    def i8(self):  return struct.unpack_from("<b", self.b, self._adv(1))[0]
    def u16(self): return struct.unpack_from("<H", self.b, self._adv(2))[0]
    def i16(self): return struct.unpack_from("<h", self.b, self._adv(2))[0]
    def u32(self): return struct.unpack_from("<I", self.b, self._adv(4))[0]
    def i32(self): return struct.unpack_from("<i", self.b, self._adv(4))[0]
    def u64(self): return struct.unpack_from("<Q", self.b, self._adv(8))[0]
    def i64(self): return struct.unpack_from("<q", self.b, self._adv(8))[0]
    def f32(self): return struct.unpack_from("<f", self.b, self._adv(4))[0]
    def f64(self): return struct.unpack_from("<d", self.b, self._adv(8))[0]
    def boolean(self): return bool(self.u8())

    def _adv(self, n):
        self.need(n); o = self.o; self.o += n; return o

    def string(self):
        n = self.u64()
        self.need(n)
        s = self.b[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return s

    def value(self, t):
        if t == UINT8:   return self.u8()
        if t == INT8:    return self.i8()
        if t == UINT16:  return self.u16()
        if t == INT16:   return self.i16()
        if t == UINT32:  return self.u32()
        if t == INT32:   return self.i32()
        if t == FLOAT32: return self.f32()
        if t == BOOL:    return self.boolean()
        if t == STRING:  return self.string()
        if t == UINT64:  return self.u64()
        if t == INT64:   return self.i64()
        if t == FLOAT64: return self.f64()
        if t == ARRAY:
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown value type {t}")


def main(path):
    with open(path, "rb") as f:
        buf = f.read()
    r = R(buf)
    magic = r.u32()
    assert magic == GGUF_MAGIC, f"bad magic {magic:#x}"
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    print(f"# gguf v{version}  tensors={n_tensors}  kv={n_kv}")

    meta = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        val = r.value(vtype)
        meta[key] = val

    # print architecture-relevant metadata
    for k in sorted(meta):
        if any(s in k for s in ("architecture", "block_count", "nextn", "mtp",
                                 "embedding_length", "head_count", "vocab")):
            v = meta[k]
            if isinstance(v, list) and len(v) > 8:
                v = f"[{len(v)} items]"
            print(f"meta: {k} = {v}")

    print("\n# --- tensors ---")
    for _ in range(n_tensors):
        try:
            name = r.string()
            ndim = r.u32()
            dims = [r.u64() for _ in range(ndim)]
            dtype = r.u32()
            offset = r.u64()
        except EOFError:
            print("...(tensor info truncated; download more header bytes)")
            break
        print(f"{name}\t{GGML_TYPE.get(dtype, dtype)}\t{dims}\toff={offset}")


if __name__ == "__main__":
    main(sys.argv[1])
