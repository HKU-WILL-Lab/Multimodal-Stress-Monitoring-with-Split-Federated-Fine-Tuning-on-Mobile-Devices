package org.mobihoc.wellbeing.wear.capture

import org.mobihoc.wellbeing.shared.SensorSample

interface SensorSource : AutoCloseable {
    fun start()
    fun stop()
    override fun close() = stop()
}

fun interface SensorSink {
    fun accept(sample: SensorSample)
}
