package org.mobihoc.wellbeing.phone.transport

import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.WearableListenerService
import org.mobihoc.wellbeing.shared.SensorWindow
import org.mobihoc.wellbeing.shared.SensorWindowCodec
import java.util.concurrent.CopyOnWriteArraySet

class PhoneDataLayerService : WearableListenerService() {
    override fun onMessageReceived(event: MessageEvent) {
        if (event.path != SENSOR_WINDOW_PATH) return
        runCatching { SensorWindowCodec.decode(event.data) }
            .onSuccess(PendingSensorWindows::publish)
    }

    companion object {
        const val SENSOR_WINDOW_PATH = "/sensor_window/v1"
    }
}

object PendingSensorWindows {
    private val listeners = CopyOnWriteArraySet<(SensorWindow) -> Unit>()
    @Volatile private var latest: SensorWindow? = null

    fun publish(window: SensorWindow) {
        latest = window
        listeners.forEach { it(window) }
    }

    fun subscribe(listener: (SensorWindow) -> Unit): AutoCloseable {
        listeners += listener
        latest?.let(listener)
        return AutoCloseable { listeners -= listener }
    }
}
