package org.mobihoc.wellbeing.wear.capture

import org.mobihoc.wellbeing.shared.SensorSample
import org.mobihoc.wellbeing.shared.SensorWindow

class SensorWindowAccumulator(private val durationMillis: Long = 30_000) {
    private val samples = ArrayList<SensorSample>()
    private var startedAt = System.currentTimeMillis()

    @Synchronized
    fun add(sample: SensorSample) {
        samples += sample
    }

    @Synchronized
    fun flush(nowMillis: Long = System.currentTimeMillis()): SensorWindow {
        val window = SensorWindow(startedAt, nowMillis, samples.toList())
        samples.clear()
        startedAt = nowMillis
        return window
    }

    fun durationMillis(): Long = durationMillis
}
