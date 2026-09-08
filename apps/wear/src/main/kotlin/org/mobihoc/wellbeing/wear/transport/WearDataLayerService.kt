package org.mobihoc.wellbeing.wear.transport

import android.os.VibrationEffect
import android.os.VibratorManager
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.WearableListenerService
import org.json.JSONObject
import org.mobihoc.wellbeing.wear.WearRuntimeStore

class WearDataLayerService : WearableListenerService() {
    override fun onMessageReceived(event: MessageEvent) {
        if (event.path != UPDATE_PATH && event.path != ALERT_PATH) return
        val payload = runCatching { JSONObject(event.data.toString(Charsets.UTF_8)) }.getOrNull() ?: return
        WearRuntimeStore.update {
            it.copy(
                currentState = payload.optString("state", "Unknown"),
                suggestion = payload.optString("suggestion", ""),
                status = "Phone result received",
            )
        }
        if (event.path == ALERT_PATH) vibrateAlert()
    }

    private fun vibrateAlert() {
        val vibrator = getSystemService(VibratorManager::class.java).defaultVibrator
        vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 220, 100, 320), -1))
    }

    companion object {
        const val UPDATE_PATH = "/affect/update/v1"
        const val ALERT_PATH = "/affect/alert/v1"
    }
}
