package org.mobihoc.wellbeing.shared

import kotlin.test.Test
import kotlin.test.assertEquals

class SensorWindowCodecTest {
    @Test
    fun roundTripsWindow() {
        val source = SensorWindow(
            100,
            30_100,
            listOf(
                SensorSample(100, SensorKind.ACCELEROMETER, listOf(0.1f, 0.2f, 0.3f)),
                SensorSample(101, SensorKind.EDA, listOf(0.7f)),
            ),
        )
        assertEquals(source, SensorWindowCodec.decode(SensorWindowCodec.encode(source)))
    }
}
