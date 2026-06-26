#!/usr/bin/env python3
"""Compute the absolute byte ranges of the blk.32.* (MTP/nextn) tensors in the
GGUF so we know how many bytes to download, and emit an extraction plan JSON."""
import json
import struct
import sys

GGUF_MAGIC = 0x46554747
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING,
 ARRAY, UINT64, INT64, FLOAT64) = range(13)
GGML_TYPE = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16"}
# bytes per element for the non-quantized types we touch
ELEM_BYTES = {"F32": 4, "F16": 2, "BF16": 2}


class R:
    def __init__(self, b): self.b = b; self.o = 0
    def _adv(self, n):
        if self.o + n > len(self.b): raise EOFError(f"need {n} at {self.o}")
        o = self.o; self.o += n; return o
    def u8(self):  o = self._adv(1); return self.b[o]
    def i8(self):  return struct.unpack_from("<b", self.b, self._adv(1))[0]
    def u16(self): return struct.unpack_from("<H", self.b, self._adv(2))[0]
    def i16(self): return struct.unpack_from("<h", self.b, self._adv(2))[0]
    def u32(self): return struct.unpack_from("<I", self.b, self._adv(4))[0]
    def i32(self): return struct.unpack_from("<i", self.b, self._adv(4))[0]
    def u64(self): return struct.unpack_from("<Q", self.b, self._adv(8))[0]
    def i64(self): return struct.unpack_from("<q", self.b, self._adv(8))[0]
    def f32(self): return struct.unpack_from("<f", self.b, self._adv(4))[0]
    def f64(self): return struct.unpack_from("<d", self.b, self._adv(8))[0]
    def string(self):
        n = self.u64(); o = self._adv(n)
        return self.b[o:o + n].decode("utf-8", "replace")
    def value(self, t):
        if t == UINT8: return self.u8()
        if t == INT8: return self.i8()
        if t == UINT16: return self.u16()
        if t == INT16: return self.i16()
        if t == UINT32: return self.u32()
        if t == INT32: return self.i32()
        if t == FLOAT32: return self.f32()
        if t == BOOL: return bool(self.u8())
        if t == STRING: return self.string()
        if t == UINT64: return self.u64()
        if t == INT64: return self.i64()
        if t == FLOAT64: return self.f64()
        if t == ARRAY:
            et = self.u32(); n = self.u64()
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"vtype {t}")


def main(path):
    with open(path, "rb") as f:
        buf = f.read()
    r = R(buf)
    assert r.u32() == GGUF_MAGIC
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    meta = {}
    for _ in range(n_kv):
        k = r.string(); t = r.u32(); meta[k] = r.value(t)
    alignment = int(meta.get("general.alignment", 32))

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        ndim = r.u32()
        dims = [r.u64() for _ in range(ndim)]
        dtype = r.u32()
        rel_off = r.u64()
        tensors.append((name, GGML_TYPE.get(dtype, dtype), dims, rel_off))

    # data section starts after tensor-info, aligned up to `alignment`
    info_end = r.o
    data_start = (info_end + alignment - 1) // alignment * alignment

    plan = {"data_start": data_start, "alignment": alignment, "tensors": []}
    max_abs_end = 0
    for name, dt, dims, rel in tensors:
        if not name.startswith("blk.32."):
            continue
        if dt not in ELEM_BYTES:
            print(f"!! {name} is {dt}, not plain float — would need dequant", file=sys.stderr)
            nbytes = None
        else:
            nelem = 1
            for d in dims: nelem *= d
            nbytes = nelem * ELEM_BYTES[dt]
        abs_off = data_start + rel
        abs_end = abs_off + (nbytes or 0)
        max_abs_end = max(max_abs_end, abs_end)
        plan["tensors"].append({
            "name": name, "dtype": dt, "dims": dims,
            "abs_offset": abs_off, "nbytes": nbytes,
        })
    plan["download_through"] = max_abs_end
    print(json.dumps(plan, indent=2))
    print(f"\n# data_start = {data_start:,}", file=sys.stderr)
    print(f"# need to download through byte {max_abs_end:,} "
          f"(~{max_abs_end/1e6:.0f} MB)", file=sys.stderr)
    print(f"# blk.32 tensor count = {len(plan['tensors'])}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
