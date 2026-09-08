package org.mobihoc.wellbeing.phone.training

import org.json.JSONObject

enum class TrainingPhase { UNAVAILABLE, IDLE, STARTING, TRAINING, WAITING, COMPLETE, FAILED, CANCELLING }

data class TrainingStatus(
    val phase: TrainingPhase,
    val round: Int = 0,
    val totalRounds: Int = 0,
    val localStep: Int = 0,
    val totalLocalSteps: Int = 0,
    val loss: Float? = null,
    val message: String = "",
) {
    val running: Boolean
        get() = phase in setOf(
            TrainingPhase.STARTING,
            TrainingPhase.TRAINING,
            TrainingPhase.WAITING,
            TrainingPhase.CANCELLING,
        )
}

data class TrainingRequest(
    val clientId: String,
    val configPath: String,
    val modelDirectory: String,
    val encoderProgram: String,
    val datasetPath: String,
    val mainServer: String,
    val federatedServer: String,
    val encoderInputLength: Int,
)

interface TrainingEngine : AutoCloseable {
    fun available(): Boolean
    fun start(request: TrainingRequest)
    fun status(): TrainingStatus
    fun cancel()
    override fun close() = Unit
}

class NativeSflTrainingEngine private constructor() : TrainingEngine {
    override fun available(): Boolean = NativeTrainingBridge.available

    override fun start(request: TrainingRequest) {
        check(NativeTrainingBridge.start(
            request.clientId,
            request.configPath,
            request.modelDirectory,
            request.encoderProgram,
            request.datasetPath,
            request.mainServer,
            request.federatedServer,
            request.encoderInputLength,
        )) { "A training session is already active" }
    }

    override fun status(): TrainingStatus {
        val value = JSONObject(NativeTrainingBridge.status())
        return TrainingStatus(
            phase = TrainingPhase.valueOf(value.getString("phase")),
            round = value.optInt("round"),
            totalRounds = value.optInt("totalRounds"),
            localStep = value.optInt("localStep"),
            totalLocalSteps = value.optInt("totalLocalSteps"),
            loss = if (value.has("loss") && !value.isNull("loss")) value.getDouble("loss").toFloat() else null,
            message = value.optString("message"),
        )
    }

    override fun cancel() = NativeTrainingBridge.cancel()
    override fun close() = NativeTrainingBridge.close()

    companion object {
        fun createOrNull(): NativeSflTrainingEngine? =
            if (NativeTrainingBridge.available) NativeSflTrainingEngine() else null
    }
}

private object NativeTrainingBridge {
    val available: Boolean = runCatching {
        System.loadLibrary("wellbeing_sfl")
        nativeAvailable()
    }.getOrDefault(false)

    @JvmStatic private external fun nativeAvailable(): Boolean
    @JvmStatic external fun start(
        clientId: String,
        configPath: String,
        modelDirectory: String,
        encoderProgram: String,
        datasetPath: String,
        mainServer: String,
        federatedServer: String,
        encoderInputLength: Int,
    ): Boolean
    @JvmStatic external fun status(): String
    @JvmStatic external fun cancel()
    @JvmStatic external fun close()
}
