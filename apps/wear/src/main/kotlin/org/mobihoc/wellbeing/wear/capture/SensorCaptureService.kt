package org.mobihoc.wellbeing.wear.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import androidx.core.app.NotificationCompat
import org.mobihoc.wellbeing.wear.R
import org.mobihoc.wellbeing.wear.WearRuntimeStore
import org.mobihoc.wellbeing.wear.transport.WatchWindowSender

class SensorCaptureService : Service() {
    private val handler = Handler(Looper.getMainLooper())
    private val accumulator = SensorWindowAccumulator()
    private lateinit var sender: WatchWindowSender
    private var sensorSource: SensorSource? = null

    private val flushWindow = object : Runnable {
        override fun run() {
            sender.send(accumulator.flush())
            handler.postDelayed(this, accumulator.durationMillis())
        }
    }

    override fun onCreate() {
        super.onCreate()
        sender = WatchWindowSender(this)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> stopCapture()
            else -> startCapture()
        }
        return START_STICKY
    }

    private fun startCapture() {
        if (sensorSource != null) return
        startForeground(NOTIFICATION_ID, notification("Collecting a 30-second window"))
        sensorSource = SensorSourceFactory.create(
            context = this,
            sink = SensorSink(accumulator::add),
            onStatus = { message -> WearRuntimeStore.update { it.copy(status = message) } },
        ).also(SensorSource::start)
        handler.postDelayed(flushWindow, accumulator.durationMillis())
        WearRuntimeStore.update { it.copy(collecting = true, status = "Collecting sensors") }
    }

    private fun stopCapture() {
        handler.removeCallbacks(flushWindow)
        sensorSource?.close()
        sensorSource = null
        WearRuntimeStore.update { it.copy(collecting = false, status = "Stopped") }
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        handler.removeCallbacks(flushWindow)
        sensorSource?.close()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Sensor capture", NotificationManager.IMPORTANCE_LOW),
        )
    }

    private fun notification(text: String): Notification = NotificationCompat.Builder(this, CHANNEL_ID)
        .setContentTitle("MobiWellbeing")
        .setContentText(text)
        .setSmallIcon(android.R.drawable.ic_menu_compass)
        .setOngoing(true)
        .build()

    companion object {
        const val ACTION_START = "org.mobihoc.wellbeing.START_CAPTURE"
        const val ACTION_STOP = "org.mobihoc.wellbeing.STOP_CAPTURE"
        private const val CHANNEL_ID = "sensor_capture"
        private const val NOTIFICATION_ID = 1001
    }
}
