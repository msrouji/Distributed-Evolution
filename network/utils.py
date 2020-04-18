import struct

def intToBytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

def intFromBytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

def floatToBytes(x: float) -> bytes:
    return struct.pack('!d', x)

def floatFromBytes(x: bytes) -> float:
    return struct.unpack('!d', x)

def stringToBytes(x: str) -> bytes:
    return x.encode('utf-8')

def stringFromBytes(x: bytes) -> str:
    return x.decode('utf-8')