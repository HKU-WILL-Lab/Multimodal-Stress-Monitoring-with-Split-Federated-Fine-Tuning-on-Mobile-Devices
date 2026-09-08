package org.mobihoc.wellbeing.wear.transport

import android.content.Context
import com.google.android.gms.wearable.Wearable
import org.mobihoc.wellbeing.shared.SensorWindow
import org.mobihoc.wellbeing.shared.SensorWindowCodec
import org.mobihoc.wellbeing.wear.WearRuntimeStore

class WatchWindowSender(context: Context) {
    private val appContext = context.applicationContext

    fun send(window: SensorWindow) {
        val bytes = SensorWindowCodec.encode(window)
        Wearable.getNodeClient(appContext).connectedNodes
            .addOnSuccessListener { nodes ->
                if (nodes.isEmpty()) {
                    WearRuntimeStore.update { it.copy(status = "Phone disconnected") }
                } else {
                    nodes.forEach { node ->
                        Wearable.getMessageClient(appContext)
                            .sendMessage(node.id, SENSOR_WINDOW_PATH, bytes)
                            .addOnSuccessListener {
                                WearRuntimeStore.update { it.copy(status = "30 s window sent") }
                            }
                            .addOnFailureListener { error ->
                                WearRuntimeStore.update { it.copy(status = "Send failed: ${error.message}") }
                            }
                    }
                }
            }
    }

    companion object {
        const val SENSOR_WINDOW_PATH = "/sensor_window/v1"
    }
}
