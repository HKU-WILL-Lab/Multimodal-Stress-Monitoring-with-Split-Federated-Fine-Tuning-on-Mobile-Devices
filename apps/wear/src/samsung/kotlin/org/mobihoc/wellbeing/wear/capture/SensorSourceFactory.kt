package org.mobihoc.wellbeing.wear.capture

import android.content.Context
import com.samsung.android.service.health.tracking.ConnectionListener
import com.samsung.android.service.health.tracking.HealthTracker
import com.samsung.android.service.health.tracking.HealthTrackerException
import com.samsung.android.service.health.tracking.HealthTrackingService
import com.samsung.android.service.health.tracking.data.DataPoint
import com.samsung.android.service.health.tracking.data.HealthTrackerType
import com.samsung.android.service.health.tracking.data.ValueKey
import org.mobihoc.wellbeing.shared.SensorKind
import org.mobihoc.wellbeing.shared.SensorSample

object SensorSourceFactory {
    fun create(context: Context, sink: SensorSink, onStatus: (String) -> Unit): SensorSource =
        SamsungHealthSensorSource(context.applicationContext, sink, onStatus)
}

private class SamsungHealthSensorSource(
    context: Context,
    private val sink: SensorSink,
    private val onStatus: (String) -> Unit,
) : SensorSource, ConnectionListener {
    private val service = HealthTrackingService(this, context)
    private val trackers = ArrayList<HealthTracker>()

    override fun start() {
        onStatus("Connecting to Health Sensor Service")
        service.connectService()
    }

    override fun onConnectionSuccess() {
        trackers += tracker(HealthTrackerType.ACCELEROMETER_CONTINUOUS, SensorKind.ACCELEROMETER) { point ->
            val factor = (9.81 / (16383.75 / 4.0)).toFloat()
            listOf(
                point.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_X) * factor,
                point.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_Y) * factor,
                point.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_Z) * factor,
            )
        }
        trackers += tracker(HealthTrackerType.PPG_CONTINUOUS, SensorKind.PPG) { point ->
            listOf(
                point.getValue(ValueKey.PpgSet.PPG_GREEN).toFloat(),
                point.getValue(ValueKey.PpgSet.PPG_RED).toFloat(),
                point.getValue(ValueKey.PpgSet.PPG_IR).toFloat(),
            )
        }
        trackers += tracker(HealthTrackerType.EDA_CONTINUOUS, SensorKind.EDA) { point ->
            listOf(point.getValue(ValueKey.EdaSet.SKIN_CONDUCTANCE))
        }
        trackers += tracker(HealthTrackerType.SKIN_TEMPERATURE_CONTINUOUS, SensorKind.SKIN_TEMPERATURE) { point ->
            listOf(
                point.getValue(ValueKey.SkinTemperatureSet.SKIN_TEMPERATURE),
                point.getValue(ValueKey.SkinTemperatureSet.AMBIENT_TEMPERATURE),
            )
        }
        trackers += tracker(HealthTrackerType.HEART_RATE_CONTINUOUS, SensorKind.HEART_RATE) { point ->
            listOf(point.getValue(ValueKey.HeartRateSet.HEART_RATE).toFloat())
        }
        onStatus("Watch8 sensors active")
    }

    private fun tracker(
        type: HealthTrackerType,
        kind: SensorKind,
        values: (DataPoint) -> List<Float>,
    ): HealthTracker {
        val tracker = service.getHealthTracker(type)
        tracker.setEventListener(object : HealthTracker.TrackerEventListener {
            override fun onDataReceived(points: List<DataPoint>) {
                points.forEach { point ->
                    runCatching { SensorSample(point.timestamp, kind, values(point)) }
                        .onSuccess(sink::accept)
                }
            }

            override fun onFlushCompleted() = Unit

            override fun onError(error: HealthTracker.TrackerError) {
                onStatus("$kind sensor error: $error")
            }
        })
        return tracker
    }

    override fun onConnectionEnded() {
        onStatus("Health Sensor Service disconnected")
    }

    override fun onConnectionFailed(error: HealthTrackerException) {
        onStatus("Health Sensor Service failed: ${error.errorCode}")
    }

    override fun stop() {
        trackers.forEach { runCatching { it.unsetEventListener() } }
        trackers.clear()
        runCatching { service.disconnectService() }
    }
}
