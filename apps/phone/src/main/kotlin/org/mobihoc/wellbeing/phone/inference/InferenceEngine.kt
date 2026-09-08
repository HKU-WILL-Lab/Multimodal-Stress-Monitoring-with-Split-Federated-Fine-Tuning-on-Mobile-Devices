package org.mobihoc.wellbeing.phone.inference

import org.json.JSONObject
import org.mobihoc.wellbeing.shared.ModelOutput
import org.mobihoc.wellbeing.shared.SensorWindow
import org.mobihoc.wellbeing.shared.SensorWindowCodec
import java.util.concurrent.atomic.AtomicInteger

interface InferenceEngine {
    fun infer(window: SensorWindow): ModelOutput
}

class DemoInferenceEngine : InferenceEngine {
    private val index = AtomicInteger(0)
    private val sequence = listOf(
        Triple(4, 2, "Valence remained positive while arousal gradually decreased."),
        Triple(5, 5, "Both valence and arousal increased across this window."),
        Triple(2, 5, "Valence decreased while arousal stayed elevated."),
        Triple(1, 2, "Arousal and valence remained low across this window."),
        Triple(3, 3, "The signals remained near the central affective region."),
    )

    override fun infer(window: SensorWindow): ModelOutput {
        val (valence, arousal, assessment) = sequence[index.getAndIncrement().mod(sequence.size)]
        return ModelOutput(valence, arousal, assessment, window.endedAtMillis)
    }
}

class NativeSflInferenceEngine private constructor() : InferenceEngine, AutoCloseable {
    fun configure(deploymentConfig: String) = NativeSflBridge.configure(deploymentConfig)
    val configured: Boolean get() = NativeSflBridge.configured()

    override fun infer(window: SensorWindow): ModelOutput {
        val response = JSONObject(NativeSflBridge.infer(SensorWindowCodec.encode(window)))
        return ModelOutput(
            valence = response.getInt("valence"),
            arousal = response.getInt("arousal"),
            assessment = response.getString("assessment"),
            timestampMillis = window.endedAtMillis,
        )
    }

    companion object {
        fun createOrNull(): NativeSflInferenceEngine? = if (NativeSflBridge.available) {
            NativeSflInferenceEngine()
        } else {
            null
        }
    }

    override fun close() = NativeSflBridge.close()
}

private object NativeSflBridge {
    val available: Boolean = runCatching {
        System.loadLibrary("wellbeing_inference")
        nativeInferenceAvailable()
    }.getOrDefault(false)

    @JvmStatic private external fun nativeInferenceAvailable(): Boolean
    @JvmStatic external fun configure(deploymentConfig: String)
    @JvmStatic external fun configured(): Boolean
    @JvmStatic external fun infer(sensorWindow: ByteArray): String
    @JvmStatic external fun close()
}
