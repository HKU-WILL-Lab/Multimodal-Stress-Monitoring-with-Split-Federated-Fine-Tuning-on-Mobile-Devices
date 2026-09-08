package org.mobihoc.wellbeing.shared

import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class SensorKind(val wireCode: Byte) {
    ACCELEROMETER(1),
    PPG(2),
    EDA(3),
    SKIN_TEMPERATURE(4),
    HEART_RATE(5);

    companion object {
        fun fromWireCode(code: Byte): SensorKind = entries.firstOrNull { it.wireCode == code }
            ?: error("Unknown sensor kind: $code")
    }
}

data class SensorSample(
    val timestampMillis: Long,
    val kind: SensorKind,
    val values: List<Float>,
) {
    init {
        require(values.isNotEmpty() && values.size <= 8) { "A sensor sample must contain 1..8 values" }
    }
}

data class SensorWindow(
    val startedAtMillis: Long,
    val endedAtMillis: Long,
    val samples: List<SensorSample>,
) {
    init {
        require(endedAtMillis >= startedAtMillis)
    }
}

/** Compact binary format used only between the paired watch and phone. */
object SensorWindowCodec {
    private const val MAGIC = 0x4D574231 // MWB1
    private const val HEADER_BYTES = 4 + 8 + 8 + 4
    private const val MAX_SAMPLES = 10_000

    fun encode(window: SensorWindow): ByteArray {
        val size = HEADER_BYTES + window.samples.sumOf { 8 + 1 + 1 + it.values.size * 4 }
        return ByteBuffer.allocate(size).order(ByteOrder.BIG_ENDIAN).apply {
            putInt(MAGIC)
            putLong(window.startedAtMillis)
            putLong(window.endedAtMillis)
            putInt(window.samples.size)
            window.samples.forEach { sample ->
                putLong(sample.timestampMillis)
                put(sample.kind.wireCode)
                put(sample.values.size.toByte())
                sample.values.forEach(::putFloat)
            }
        }.array()
    }

    fun decode(bytes: ByteArray): SensorWindow {
        require(bytes.size >= HEADER_BYTES) { "Truncated sensor window" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        require(buffer.int == MAGIC) { "Invalid sensor window magic" }
        val start = buffer.long
        val end = buffer.long
        val count = buffer.int
        require(count in 0..MAX_SAMPLES) { "Invalid sample count: $count" }
        val samples = ArrayList<SensorSample>(count)
        repeat(count) {
            require(buffer.remaining() >= 10) { "Truncated sample header" }
            val timestamp = buffer.long
            val kind = SensorKind.fromWireCode(buffer.get())
            val valueCount = buffer.get().toInt() and 0xFF
            require(valueCount in 1..8 && buffer.remaining() >= valueCount * 4) { "Invalid sample payload" }
            samples += SensorSample(timestamp, kind, List(valueCount) { buffer.float })
        }
        require(!buffer.hasRemaining()) { "Unexpected bytes after sensor window" }
        return SensorWindow(start, end, samples)
    }
}
