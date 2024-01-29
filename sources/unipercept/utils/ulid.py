"""
Simple implementation of ULIDs.

"""

from __future__ import annotations

import abc
import array
import binascii
import datetime
import os
import time
import typing as T
import uuid

import typing_extensions as TX

from unipercept.utils.typings import Buffer, Datetime, Primitive

__all__ = ["Timestamp", "Randomness", "ULID", "base32_encode", "base32_decode"]


class _MemView:
    """
    Wraps a buffer object, typically :class:`~bytes`, with a :class:`~memoryview` and provides easy
    type comparisons and conversions between presentation formats.
    """

    __slots__ = ["memory"]

    def __init__(self, buffer: Buffer) -> None:
        self.memory = memoryview(buffer)

    @TX.override
    def __eq__(self, other: MemViewPrimitive) -> bool:
        if isinstance(other, _MemView):
            return self.memory == other.memory
        if isinstance(other, (bytes, bytearray, memoryview)):
            return self.memory == other
        if isinstance(other, int):
            return self.int == other
        if isinstance(other, float):
            return self.float == other
        if isinstance(other, str):
            return self.str == other
        return NotImplemented

    @TX.override
    def __ne__(self, other: MemViewPrimitive) -> bool:
        if isinstance(other, _MemView):
            return self.memory != other.memory
        if isinstance(other, (bytes, bytearray, memoryview)):
            return self.memory != other
        if isinstance(other, int):
            return self.int != other
        if isinstance(other, float):
            return self.float != other
        if isinstance(other, str):
            return self.str != other
        return NotImplemented

    def __lt__(self, other: MemViewPrimitive) -> bool:
        if isinstance(other, _MemView):
            return self.int < other.int
        if isinstance(other, (bytes, bytearray)):
            return self.bytes < other
        if isinstance(other, memoryview):
            return self.bytes < other.tobytes()
        if isinstance(other, int):
            return self.int < other
        if isinstance(other, float):
            return self.float < other
        if isinstance(other, str):
            return self.str < other
        return NotImplemented

    def __gt__(self, other: MemViewPrimitive) -> bool:
        if isinstance(other, _MemView):
            return self.int > other.int
        if isinstance(other, (bytes, bytearray)):
            return self.bytes > other
        if isinstance(other, memoryview):
            return self.bytes > other.tobytes()
        if isinstance(other, int):
            return self.int > other
        if isinstance(other, float):
            return self.float > other
        if isinstance(other, str):
            return self.str > other
        return NotImplemented

    def __le__(self, other: MemViewPrimitive) -> bool:
        if isinstance(other, _MemView):
            return self.int <= other.int
        if isinstance(other, (bytes, bytearray)):
            return self.bytes <= other
        if isinstance(other, memoryview):
            return self.bytes <= other.tobytes()
        if isinstance(other, int):
            return self.int <= other
        if isinstance(other, float):
            return self.float <= other
        if isinstance(other, str):
            return self.str <= other
        return NotImplemented

    def __ge__(self, other: MemViewPrimitive) -> bool:
        if isinstance(other, _MemView):
            return self.int >= other.int
        if isinstance(other, (bytes, bytearray)):
            return self.bytes >= other
        if isinstance(other, memoryview):
            return self.bytes >= other.tobytes()
        if isinstance(other, int):
            return self.int >= other
        if isinstance(other, float):
            return self.float >= other
        if isinstance(other, str):
            return self.str >= other
        return NotImplemented

    @TX.override
    def __hash__(self) -> int:
        return hash(self.memory)

    def __bytes__(self) -> bytes:
        return self.bytes

    def __float__(self) -> float:
        return self.float

    def __int__(self) -> int:
        return self.int

    def __index__(self) -> int:
        return self.int

    @TX.override
    def __repr__(self) -> str:
        return "<{}({!r})>".format(self.__class__.__name__, str(self))

    @TX.override
    def __str__(self) -> str:
        return self.str

    @TX.override
    def __getstate__(self) -> str:
        return self.str

    def __setstate__(self, state: str) -> None:
        self.memory = memoryview(base32_decode(state))

    @property
    def bin(self) -> str:
        return bin(self.int)

    @property
    def bytes(self) -> bytes:
        return self.memory.tobytes()

    @property
    def float(self) -> float:
        return float(self.int)

    @property
    def hex(self) -> str:
        return "0x" + binascii.hexlify(self.bytes).decode()

    @property
    def int(self) -> int:
        return int.from_bytes(self.memory, byteorder="big")

    @property
    def oct(self) -> str:
        return oct(self.int)

    @property
    @abc.abstractmethod
    def str(self) -> str:
        return NotImplemented


MemViewPrimitive = _MemView | Primitive


class Timestamp(_MemView):
    """
    Represents the timestamp portion of a ULID.

    * Unix time (time since epoch) in milliseconds.
    * First 48 bits of ULID when in binary format.
    * First 10 characters of ULID when in string format.
    """

    __slots__ = _MemView.__slots__

    @classmethod
    def generate(cls) -> T.Self:
        return cls.create(int(time.time() * 1000).to_bytes(6, "big"))

    @classmethod
    def create(cls, timestamp: T.Self | Primitive | Datetime | ULID) -> T.Self:
        """
        Create a new :class:`~ulid.ulid.ULID` instance using a timestamp value of a supported type.

        The following types are supported for timestamp values:

        * :class:`~datetime.datetime`
        * :class:`~int`
        * :class:`~float`
        * :class:`~str`
        * :class:`~memoryview`
        * :class:`~ulid.ulid.Timestamp`
        * :class:`~ulid.ulid.ULID`
        * :class:`~bytes`
        * :class:`~bytearray`

        Parameters
        ----------
        timestamp : datetime, int, float, str, memoryview, Timestamp, ULID, bytes, bytearray
            Unix timestamp in seconds

        Returns
        -------
        :class:`~ulid.ulid.ULID`
            ULID using given timestamp and new randomness

        Raises
        ------
        ValueError
            When the value is an unsupported type
        ValueError
            When the value is a string and cannot be Base32 decoded
        ValueError
            When the value is or was converted to something 48 bits
        """
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()
        if isinstance(timestamp, (int, float)):
            timestamp = int(timestamp * 1000.0).to_bytes(6, byteorder="big")
        elif isinstance(timestamp, str):
            timestamp = _base32_decode_timestamp(timestamp)
        elif isinstance(timestamp, memoryview):
            timestamp = timestamp.tobytes()
        elif isinstance(timestamp, cls):
            timestamp = timestamp.bytes
        elif isinstance(timestamp, ULID):
            timestamp = timestamp.timestamp().bytes

        if not isinstance(timestamp, (bytes, bytearray)):
            msg = (
                "Expected datetime, int, float, str, memoryview, Timestamp, ULID, "
                "bytes, or bytearray; got %s"
            )
            raise ValueError(msg, type(timestamp).__name__)

        length = len(timestamp)
        if length != 6:
            msg = "Expects timestamp to be 48 bits; got {} bytes".format(length)
            raise ValueError(msg)

        return cls(timestamp)

    @property
    @TX.override
    def str(self) -> str:
        """
        Computes the string value of the timestamp from the underlying :class:`~memoryview` in Base32 encoding.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Timestamp in Base32 string form.

        Raises
        ------
        ValueError
            If underlying :class:`~memoryview` cannot be encoded.
        """
        return _base32_encode_timestamp(self.memory)

    @property
    def timestamp(self) -> float:
        """
        Computes the Unix time (seconds since epoch) from its :class:`~memoryview`.

        Returns
        -------
        float
            Timestamp in Unix time (seconds since epoch) form.
        """
        return self.int / 1000.0

    @property
    def datetime(self) -> Datetime:
        """
        Creates a :class:`~datetime.datetime` instance (assumes UTC) from the Unix time value of the timestamp
        with millisecond precision.

        Returns
        -------
        :class:`~datetime.datetime`
            Timestamp in datetime form.
        """
        milli = self.int
        micro = milli % 1000 * 1000
        sec = milli // 1000.0
        timezone = datetime.timezone.utc

        return datetime.datetime.utcfromtimestamp(sec).replace(
            microsecond=micro, tzinfo=timezone
        )

    @property
    @TX.override
    def str(self) -> str:
        return _base32_encode_timestamp(self.memory)


class Randomness(_MemView):
    """
    Represents the randomness portion of a ULID.

    * Cryptographically secure random values.
    * Last 80 bits of ULID when in binary format.
    * Last 16 characters of ULID when in string format.
    """

    __slots__ = _MemView.__slots__

    @classmethod
    def generate(cls) -> T.Self:
        return cls.create(os.urandom(10))

    @property
    @TX.override
    def str(self) -> str:
        """
        Computes the string value of the randomness from the underlying :class:`~memoryview` in Base32 encoding.

        Returns
        -------
        str
            Timestamp in Base32 string form.

        Raises
        ------
        ValueError
            If underlying :class:`~memoryview` cannot be encoded
        """
        return _base32_encode_timestamp(self.memory)

    @classmethod
    def create(cls, randomness: T.Self | Primitive) -> T.Self:
        """
        Create a new :class:`~ulid.ulid.Randomness` instance using the given randomness value of a supported type.

        The following types are supported for randomness values:

        * :class:`~int`
        * :class:`~float`
        * :class:`~str`
        * :class:`~memoryview`
        * :class:`~ulid.ulid.Randomness`
        * :class:`~ulid.ulid.ULID`
        * :class:`~bytes`
        * :class:`~bytearray`

        Parameters
        ----------
        randomness : int, float, str, memoryview, Randomness, ULID, bytes, bytearray
            Random bytes

        Returns
        -------
        :class:`~ulid.ulid.ULID`
            ULID using new timestamp and given randomness

        Raises
        ------
        ValueError
            When the value is an unsupported type
        ValueError
            When the value is a string and cannot be Base32 decoded
        ValueError
            When the value is or was converted to something 80 bits
        """
        if isinstance(randomness, (int, float)):
            randomness = int(randomness).to_bytes(10, byteorder="big")
        elif isinstance(randomness, str):
            randomness = _base32_decode_randomness(randomness)
        elif isinstance(randomness, memoryview):
            randomness = randomness.tobytes()
        elif isinstance(randomness, Randomness):
            randomness = randomness.bytes
        elif isinstance(randomness, ULID):
            randomness = randomness.randomness().bytes

        if not isinstance(randomness, (bytes, bytearray)):
            msg = (
                "Expected int, float, str, memoryview, Randomness, ULID, "
                "bytes, or bytearray; got {}".format(type(randomness).__name__)
            )
            raise ValueError(msg)

        length = len(randomness)
        if length != 10:
            msg = "Expected randomness to be 80 bits; got %d bytes"
            raise ValueError(msg, length)

        return cls(randomness)

    @property
    @TX.override
    def str(self) -> str:
        return _base32_encode_timestamp(self.memory)


class ULID(_MemView):
    """
    Represents a ULID.

    * 128 bits in binary format.
    * 26 characters in string format.
    * 16 octets.
    * Network byte order, big-endian, most significant bit first.
    """

    __slots__ = _MemView.__slots__

    @classmethod
    def generate(cls) -> T.Self:
        """
        Generate a new :class:`~ulid.ulid.ULID` instance using the current time and cryptographically secure
        randomness.

        Returns
        -------
        :class:`~ulid.ulid.ULID`
            New ULID instance
        """
        timestamp = Timestamp.generate()
        randomness = Randomness.generate()

        return cls.create(timestamp, randomness)

    @classmethod
    def create(
        cls,
        timestamp: Timestamp | Primitive | Datetime | ULID,
        randomness: Randomness | Primitive,
    ) -> T.Self:
        if not isinstance(timestamp, Timestamp):
            timestamp = Timestamp.create(timestamp)
        if not isinstance(randomness, Randomness):
            randomness = Randomness.create(randomness)

        ulid = timestamp.bytes + randomness.bytes

        return cls(ulid)

    @property
    @TX.override
    def str(self) -> str:
        """
        Computes the string value of the ULID from its :class:`~memoryview` in Base32 encoding.

        Returns
        -------
        str
            ULID in Base32 string form.

        Raises
        ------
        ValueError
            If underlying :class:`~memoryview` cannot be encoded.
        """
        return _base32_encode_ulid(self.memory)

    def timestamp(self) -> Timestamp:
        """
        Creates a :class:`~ulid.ulid.Timestamp` instance that maps to the first 48 bits of this ULID.

        Returns
        -------
        :class:`~ulid.ulid.Timestamp`
            Timestamp from first 48 bits.
        """
        return Timestamp(self.memory[:6])

    def randomness(self) -> Randomness:
        """
        Creates a :class:`~ulid.ulid.Randomness` instance that maps to the last 80 bits of this ULID.

        Returns
        -------
        :class:`~ulid.ulid.Timestamp`
            Timestamp from first 48 bits.
        """
        return Randomness(self.memory[6:])

    @property
    def uuid(self) -> uuid.UUID:
        """
        Creates a :class:`~uuid.UUID` instance of the ULID from its :class:`~bytes` representation.

        Returns
        -------
        :class:`~uuid.UUID`
            UUIDv4 from the ULID bytes
        """
        return uuid.UUID(bytes=self.bytes)

    @property
    @TX.override
    def str(self) -> str:
        return _base32_encode_ulid(self.memory)


BASE32_CHARSET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # NOTE: Exclude 'I', 'L', 'O', 'U'
BASE32_CHARMAP = array.array(
    "B",
    (
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0x00,
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x07,
        0x08,
        0x09,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x0E,
        0x0F,
        0x10,
        0x11,
        0x01,
        0x12,
        0x13,
        0x01,
        0x14,
        0x15,
        0x00,
        0x16,
        0x17,
        0x18,
        0x19,
        0x1A,
        0xFF,
        0x1B,
        0x1C,
        0x1D,
        0x1E,
        0x1F,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x0E,
        0x0F,
        0x10,
        0x11,
        0x01,
        0x12,
        0x13,
        0x01,
        0x14,
        0x15,
        0x00,
        0x16,
        0x17,
        0x18,
        0x19,
        0x1A,
        0xFF,
        0x1B,
        0x1C,
        0x1D,
        0x1E,
        0x1F,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
    ),
)


def base32_encode(value: Buffer) -> str:
    length = len(value)

    # Order here is based on assumed hot path.
    if length == 16:
        return _base32_encode_ulid(value)
    if length == 6:
        return _base32_encode_timestamp(value)
    if length == 10:
        return _base32_encode_timestamp(value)

    msg = "Expects bytes in sizes of 6, 10, or 16; got %d"
    raise ValueError(msg, length)


def _base32_encode_ulid(value: Buffer) -> str:
    length = len(value)
    if length != 16:
        msg = "Expects 16 bytes for timestamp + randomness; got %d"
        raise ValueError(msg, length)

    encoding = BASE32_CHARSET

    return (
        encoding[(value[0] & 224) >> 5]
        + encoding[value[0] & 31]
        + encoding[(value[1] & 248) >> 3]
        + encoding[((value[1] & 7) << 2) | ((value[2] & 192) >> 6)]
        + encoding[((value[2] & 62) >> 1)]
        + encoding[((value[2] & 1) << 4) | ((value[3] & 240) >> 4)]
        + encoding[((value[3] & 15) << 1) | ((value[4] & 128) >> 7)]
        + encoding[(value[4] & 124) >> 2]
        + encoding[((value[4] & 3) << 3) | ((value[5] & 224) >> 5)]
        + encoding[value[5] & 31]
        + encoding[(value[6] & 248) >> 3]
        + encoding[((value[6] & 7) << 2) | ((value[7] & 192) >> 6)]
        + encoding[(value[7] & 62) >> 1]
        + encoding[((value[7] & 1) << 4) | ((value[8] & 240) >> 4)]
        + encoding[((value[8] & 15) << 1) | ((value[9] & 128) >> 7)]
        + encoding[(value[9] & 124) >> 2]
        + encoding[((value[9] & 3) << 3) | ((value[10] & 224) >> 5)]
        + encoding[value[10] & 31]
        + encoding[(value[11] & 248) >> 3]
        + encoding[((value[11] & 7) << 2) | ((value[12] & 192) >> 6)]
        + encoding[(value[12] & 62) >> 1]
        + encoding[((value[12] & 1) << 4) | ((value[13] & 240) >> 4)]
        + encoding[((value[13] & 15) << 1) | ((value[14] & 128) >> 7)]
        + encoding[(value[14] & 124) >> 2]
        + encoding[((value[14] & 3) << 3) | ((value[15] & 224) >> 5)]
        + encoding[value[15] & 31]
    )


def _base32_encode_timestamp(timestamp: Buffer) -> str:
    length = len(timestamp)
    if length != 6:
        msg = "Expects 6 bytes for timestamp; got %d"
        raise ValueError(msg, length)

    encoding = BASE32_CHARSET

    return (
        encoding[(timestamp[0] & 224) >> 5]
        + encoding[timestamp[0] & 31]
        + encoding[(timestamp[1] & 248) >> 3]
        + encoding[((timestamp[1] & 7) << 2) | ((timestamp[2] & 192) >> 6)]
        + encoding[((timestamp[2] & 62) >> 1)]
        + encoding[((timestamp[2] & 1) << 4) | ((timestamp[3] & 240) >> 4)]
        + encoding[((timestamp[3] & 15) << 1) | ((timestamp[4] & 128) >> 7)]
        + encoding[(timestamp[4] & 124) >> 2]
        + encoding[((timestamp[4] & 3) << 3) | ((timestamp[5] & 224) >> 5)]
        + encoding[timestamp[5] & 31]
    )


def _base32_encode_timestamp(randomness: Buffer) -> str:
    length = len(randomness)
    if length != 10:
        msg = "Expects 10 bytes for randomness; got %d"
        raise ValueError(msg, length)

    encoding = BASE32_CHARSET

    return (
        encoding[(randomness[0] & 248) >> 3]
        + encoding[((randomness[0] & 7) << 2) | ((randomness[1] & 192) >> 6)]
        + encoding[(randomness[1] & 62) >> 1]
        + encoding[((randomness[1] & 1) << 4) | ((randomness[2] & 240) >> 4)]
        + encoding[((randomness[2] & 15) << 1) | ((randomness[3] & 128) >> 7)]
        + encoding[(randomness[3] & 124) >> 2]
        + encoding[((randomness[3] & 3) << 3) | ((randomness[4] & 224) >> 5)]
        + encoding[randomness[4] & 31]
        + encoding[(randomness[5] & 248) >> 3]
        + encoding[((randomness[5] & 7) << 2) | ((randomness[6] & 192) >> 6)]
        + encoding[(randomness[6] & 62) >> 1]
        + encoding[((randomness[6] & 1) << 4) | ((randomness[7] & 240) >> 4)]
        + encoding[((randomness[7] & 15) << 1) | ((randomness[8] & 128) >> 7)]
        + encoding[(randomness[8] & 124) >> 2]
        + encoding[((randomness[8] & 3) << 3) | ((randomness[9] & 224) >> 5)]
        + encoding[randomness[9] & 31]
    )


def base32_decode(value: str) -> bytes:
    length = len(value)

    # Order here is based on assumed hot path.
    if length == 26:
        return _base32_decode_ulid(value)
    if length == 10:
        return _base32_decode_timestamp(value)
    if length == 16:
        return _base32_decode_randomness(value)

    msg = "Expects string in lengths of 10, 16, or 26; got %d"
    raise ValueError(msg, length)


def _base32_decode_ulid(value: str) -> bytes:
    encoded = _base32_as_bytes(value, 26)

    decoding = BASE32_CHARMAP

    return bytes(
        (
            ((decoding[encoded[0]] << 5) | decoding[encoded[1]]) & 0xFF,
            ((decoding[encoded[2]] << 3) | (decoding[encoded[3]] >> 2)) & 0xFF,
            (
                (decoding[encoded[3]] << 6)
                | (decoding[encoded[4]] << 1)
                | (decoding[encoded[5]] >> 4)
            )
            & 0xFF,
            ((decoding[encoded[5]] << 4) | (decoding[encoded[6]] >> 1)) & 0xFF,
            (
                (decoding[encoded[6]] << 7)
                | (decoding[encoded[7]] << 2)
                | (decoding[encoded[8]] >> 3)
            )
            & 0xFF,
            ((decoding[encoded[8]] << 5) | (decoding[encoded[9]])) & 0xFF,
            ((decoding[encoded[10]] << 3) | (decoding[encoded[11]] >> 2)) & 0xFF,
            (
                (decoding[encoded[11]] << 6)
                | (decoding[encoded[12]] << 1)
                | (decoding[encoded[13]] >> 4)
            )
            & 0xFF,
            ((decoding[encoded[13]] << 4) | (decoding[encoded[14]] >> 1)) & 0xFF,
            (
                (decoding[encoded[14]] << 7)
                | (decoding[encoded[15]] << 2)
                | (decoding[encoded[16]] >> 3)
            )
            & 0xFF,
            ((decoding[encoded[16]] << 5) | (decoding[encoded[17]])) & 0xFF,
            ((decoding[encoded[18]] << 3) | (decoding[encoded[19]] >> 2)) & 0xFF,
            (
                (decoding[encoded[19]] << 6)
                | (decoding[encoded[20]] << 1)
                | (decoding[encoded[21]] >> 4)
            )
            & 0xFF,
            ((decoding[encoded[21]] << 4) | (decoding[encoded[22]] >> 1)) & 0xFF,
            (
                (decoding[encoded[22]] << 7)
                | (decoding[encoded[23]] << 2)
                | (decoding[encoded[24]] >> 3)
            )
            & 0xFF,
            ((decoding[encoded[24]] << 5) | (decoding[encoded[25]])) & 0xFF,
        )
    )


def _base32_decode_timestamp(timestamp: str) -> bytes:
    encoded = _base32_as_bytes(timestamp, 10)

    decoding = BASE32_CHARMAP

    return bytes(
        (
            ((decoding[encoded[0]] << 5) | decoding[encoded[1]]) & 0xFF,
            ((decoding[encoded[2]] << 3) | (decoding[encoded[3]] >> 2)) & 0xFF,
            (
                (decoding[encoded[3]] << 6)
                | (decoding[encoded[4]] << 1)
                | (decoding[encoded[5]] >> 4)
            )
            & 0xFF,
            ((decoding[encoded[5]] << 4) | (decoding[encoded[6]] >> 1)) & 0xFF,
            (
                (decoding[encoded[6]] << 7)
                | (decoding[encoded[7]] << 2)
                | (decoding[encoded[8]] >> 3)
            )
            & 0xFF,
            ((decoding[encoded[8]] << 5) | (decoding[encoded[9]])) & 0xFF,
        )
    )


def _base32_decode_randomness(randomness: str) -> bytes:
    encoded = _base32_as_bytes(randomness, 16)

    decoding = BASE32_CHARMAP

    return bytes(
        (
            ((decoding[encoded[0]] << 3) | (decoding[encoded[1]] >> 2)) & 0xFF,
            (
                (decoding[encoded[1]] << 6)
                | (decoding[encoded[2]] << 1)
                | (decoding[encoded[3]] >> 4)
            )
            & 0xFF,
            ((decoding[encoded[3]] << 4) | (decoding[encoded[4]] >> 1)) & 0xFF,
            (
                (decoding[encoded[4]] << 7)
                | (decoding[encoded[5]] << 2)
                | (decoding[encoded[6]] >> 3)
            )
            & 0xFF,
            ((decoding[encoded[6]] << 5) | (decoding[encoded[7]])) & 0xFF,
            ((decoding[encoded[8]] << 3) | (decoding[encoded[9]] >> 2)) & 0xFF,
            (
                (decoding[encoded[9]] << 6)
                | (decoding[encoded[10]] << 1)
                | (decoding[encoded[11]] >> 4)
            )
            & 0xFF,
            ((decoding[encoded[11]] << 4) | (decoding[encoded[12]] >> 1)) & 0xFF,
            (
                (decoding[encoded[12]] << 7)
                | (decoding[encoded[13]] << 2)
                | (decoding[encoded[14]] >> 3)
            )
            & 0xFF,
            ((decoding[encoded[14]] << 5) | (decoding[encoded[15]])) & 0xFF,
        )
    )


def _base32_as_bytes(value: str, expected_length: int) -> bytes:
    """
    Convert the given string to bytes and validate it is within the Base32 character set.

    Parameters
    ----------
    value : str
        String to convert to bytes
    expected_length : int
        Expected length of the input string

    Returns
    -------
    bytes
        Value converted to bytes.
    """
    length = len(value)
    if length != expected_length:
        msg = "Expects %d characters for decoding; got %d"
        raise ValueError(msg, expected_length, length)

    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as ex:
        msg = "Expects value that can be encoded in ASCII charset: %s"
        raise ValueError(msg, ex) from ex

    decoding = BASE32_CHARMAP

    # Confirm all bytes are valid Base32 decode characters.
    # Note: ASCII encoding handles the out of range checking for us.
    for byte in encoded:
        if decoding[byte] > 31:
            msg = f"Non-base32 character found: {chr(byte)!r}"
            raise ValueError(msg)

    # Confirm most significant bit on timestamp value is limited so it can be stored in 128-bits.
    if length in (10, 26):
        msb = decoding[encoded[0]]
        if msb > 7:
            msg = (
                "Timestamp value too large and will overflow 128-bits. "
                'Must be between b"0" and b"7"'
            )
            raise ValueError(msg)

    return encoded
