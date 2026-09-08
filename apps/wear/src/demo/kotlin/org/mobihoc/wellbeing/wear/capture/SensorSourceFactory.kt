package org.mobihoc.wellbeing.wear.capture

import android.content.Context
import android.os.Handler
import android.os.Looper
import org.mobihoc.wellbeing.shared.SensorKind
import org.mobihoc.wellbeing.shared.SensorSample
import kotlin.math.sin

object SensorSourceFactory {
    fun create(context: Context, sink: SensorSink, onStatus: (String) -> Unit): SensorSource =
        DemoSensorSource(sink, onStatus)
}

private class DemoSensorSource(
    private val sink: SensorSink,
    private val onStatus: (String) -> Unit,
) : SensorSource {
    private val handler = Handler(Looper.getMainLooper())
    private var tick = 0
    private val sample = object : Runnable {
        override fun run() {
            val now = System.currentTimeMillis()
            val wave = sin(tick++ / 4.0).toFloat()
            sink.accept(SensorSample(now, SensorKind.ACCELEROMETER, listOf(wave, wave / 2, 9.81f)))
            sink.accept(SensorSample(now, SensorKind.PPG, listOf(50_000f + wave * 2_000, 30_000f, 25_000f)))
            sink.accept(SensorSample(now, SensorKind.EDA, listOf(0.8f + wave * 0.1f)))
            sink.accept(SensorSample(now, SensorKind.SKIN_TEMPERATURE, listOf(32.1f, 25.0f)))
            sink.accept(SensorSample(now, SensorKind.HEART_RATE, listOf(72f + wave * 3)))
            handler.postDelayed(this, 1_000)
        }
    }

    override fun start() {
        onStatus("Synthetic Watch8 signals")
        handler.post(sample)
    }

    override fun stop() {
        handler.removeCallbacks(sample)
    }
}
