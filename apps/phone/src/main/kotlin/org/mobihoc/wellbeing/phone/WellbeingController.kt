package org.mobihoc.wellbeing.phone

import android.content.Context
import android.os.Handler
import android.os.Looper
import org.mobihoc.wellbeing.phone.inference.DemoInferenceEngine
import org.mobihoc.wellbeing.phone.inference.InferenceEngine
import org.mobihoc.wellbeing.phone.inference.NativeSflInferenceEngine
import org.mobihoc.wellbeing.phone.transport.PendingSensorWindows
import org.mobihoc.wellbeing.phone.transport.WatchMessenger
import org.mobihoc.wellbeing.shared.ModelOutput
import org.mobihoc.wellbeing.shared.SensorWindow
import org.mobihoc.wellbeing.shared.TrendFormatter
import org.mobihoc.wellbeing.phone.training.NativeSflTrainingEngine
import org.mobihoc.wellbeing.phone.training.TrainingPhase
import org.mobihoc.wellbeing.phone.training.TrainingRequest
import org.mobihoc.wellbeing.phone.training.TrainingStatus
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

data class PhoneRuntimeState(
    val outputs: List<ModelOutput> = emptyList(),
    val latestWindow: SensorWindow? = null,
    val isProcessing: Boolean = false,
    val source: String = "Waiting for Watch8",
    val error: String? = null,
    val training: TrainingStatus = TrainingStatus(TrainingPhase.UNAVAILABLE, message = "Native training runtime is not installed"),
    val inferenceRuntimeAvailable: Boolean = false,
    val inferenceConfigured: Boolean = false,
    val inferenceMessage: String = "Native inference runtime is not installed",
) {
    val current: ModelOutput? get() = outputs.lastOrNull()
    val trend: String get() = TrendFormatter.format(outputs)
}

class WellbeingController(context: Context) : AutoCloseable {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadScheduledExecutor()
    private val demoEngine = DemoInferenceEngine()
    private val nativeInference = NativeSflInferenceEngine.createOrNull()
    private val watchMessenger = WatchMessenger(context)
    private val subscription = PendingSensorWindows.subscribe(::process)
    private val trainingEngine = NativeSflTrainingEngine.createOrNull()
    private var trainingPoll: ScheduledFuture<*>? = null

    @Volatile
    var state = PhoneRuntimeState(
        source = "Waiting for Watch8",
        inferenceRuntimeAvailable = nativeInference != null,
        inferenceConfigured = nativeInference?.configured == true,
        inferenceMessage = if (nativeInference == null) "Native inference runtime is not installed" else "Select an inference deployment config",
    )
        private set

    init {
        if (trainingEngine != null) {
            state = state.copy(training = TrainingStatus(TrainingPhase.IDLE, message = "Ready"))
        }
    }

    fun startTraining(request: TrainingRequest) {
        val training = trainingEngine ?: run {
            state = state.copy(error = "Install the wellbeing_sfl native runtime before starting training")
            return
        }
        worker.execute {
            runCatching { training.start(request) }
                .onSuccess {
                    main.post { state = state.copy(training = training.status(), error = null) }
                    trainingPoll?.cancel(false)
                    trainingPoll = worker.scheduleAtFixedRate({
                        runCatching { training.status() }
                            .onSuccess { status -> main.post { state = state.copy(training = status) } }
                            .onFailure { error -> main.post { state = state.copy(error = error.message) } }
                    }, 1, 1, TimeUnit.SECONDS)
                }
                .onFailure { error -> main.post { state = state.copy(error = error.message) } }
        }
    }

    fun cancelTraining() {
        trainingEngine?.cancel()
    }

    fun configureInference(deploymentConfig: String) {
        val inference = nativeInference ?: run {
            state = state.copy(error = "Install the wellbeing_inference native runtime before loading inference")
            return
        }
        main.post { state = state.copy(isProcessing = true, error = null, inferenceMessage = "Loading local model…") }
        worker.execute {
            runCatching { inference.configure(deploymentConfig) }
                .onSuccess {
                    main.post {
                        state = state.copy(
                            isProcessing = false,
                            inferenceConfigured = true,
                            inferenceMessage = "Local Llama inference is ready",
                            source = "On-device inference",
                        )
                    }
                }
                .onFailure { error ->
                    main.post {
                        state = state.copy(
                            isProcessing = false,
                            inferenceConfigured = false,
                            inferenceMessage = "Model loading failed",
                            error = error.message,
                        )
                    }
                }
        }
    }

    fun simulateNextWindow() {
        val now = System.currentTimeMillis()
        process(SensorWindow(now - 30_000, now, emptyList()))
    }

    private fun process(window: SensorWindow) {
        main.post { state = state.copy(isProcessing = true, error = null) }
        worker.execute {
            val selected = if (window.samples.isEmpty()) {
                demoEngine
            } else {
                nativeInference?.takeIf { it.configured }
                    ?: run {
                        main.post { state = state.copy(isProcessing = false, error = "Load the local inference deployment before processing watch data") }
                        return@execute
                    }
            }
            runCatching { selected.infer(window) }
                .onSuccess { output ->
                    main.post {
                        state = state.copy(
                            outputs = (state.outputs + output).takeLast(3),
                            latestWindow = window.takeIf { it.samples.isNotEmpty() },
                            isProcessing = false,
                            source = if (window.samples.isEmpty()) "Demo inference" else "On-device Llama · Watch8 window",
                        )
                        watchMessenger.publish(output)
                    }
                }
                .onFailure { error ->
                    main.post { state = state.copy(isProcessing = false, error = error.message) }
                }
        }
    }

    override fun close() {
        subscription.close()
        trainingPoll?.cancel(true)
        trainingEngine?.close()
        nativeInference?.close()
        worker.shutdownNow()
    }
}
