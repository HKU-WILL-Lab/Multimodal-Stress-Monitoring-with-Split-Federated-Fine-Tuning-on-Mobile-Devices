package org.mobihoc.wellbeing.phone.transport

import android.content.Context
import com.google.android.gms.wearable.Wearable
import org.json.JSONObject
import org.mobihoc.wellbeing.shared.ModelOutput

class WatchMessenger(context: Context) {
    private val appContext = context.applicationContext

    fun publish(output: ModelOutput) {
        val payload = JSONObject()
            .put("state", output.state.displayName)
            .put("valence", output.valence)
            .put("arousal", output.arousal)
            .put("suggestion", output.state.suggestion)
            .toString()
            .toByteArray(Charsets.UTF_8)
        val path = if (output.state.shouldAlert) ALERT_PATH else UPDATE_PATH

        Wearable.getNodeClient(appContext).connectedNodes.addOnSuccessListener { nodes ->
            nodes.forEach { node ->
                Wearable.getMessageClient(appContext).sendMessage(node.id, path, payload)
            }
        }
    }

    companion object {
        const val UPDATE_PATH = "/affect/update/v1"
        const val ALERT_PATH = "/affect/alert/v1"
    }
}
