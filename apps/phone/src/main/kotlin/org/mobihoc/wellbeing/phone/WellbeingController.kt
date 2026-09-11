package org.mobihoc.wellbeing.phone

import android.content.Context
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import org.mobihoc.wellbeing.phone.inference.DemoInferenceEngine
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
import org.mobihoc.wellbeing.phone.training.TrainingMetricsReader
import org.mobihoc.wellbeing.phone.training.readAggregationStatus
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

data class PhoneUiState(
    val outputs: List<ModelOutput> = emptyList(),
    val latestWindow: SensorWindow? = null,
    val isProcessing: Boolean = false,
    val source: String = "Waiting for Watch8",
    val error: String? = null,
    val training: TrainingStatus = TrainingStatus(TrainingPhase.UNAVAILABLE, message = "Native training runtime is not installed"),
    val inferenceRuntimeAvailable: Boolean = false,
    val inferenceConfigured: Boolean = false,
    val inferenceMessage: String = "Native inference runtime is not installed",
    val cuttingLayer: Int? = null,
    val trainingHistory: List<TrainingStatus> = emptyList(),
    val aggregation: JSONObject = JSONObject().put("phase", "OFFLINE"),
) {
    val current: ModelOutput? get() = outputs.lastOrNull()
    val trend: String get() = TrendFormatter.format(outputs)
}

class WellbeingController(context: Context) : AutoCloseable {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadScheduledExecutor()
    private val statusWorker = Executors.newSingleThreadScheduledExecutor()
    private val preferences = context.getSharedPreferences("stitch-settings", Context.MODE_PRIVATE)
    @Volatile private var activeRunId: String? = null
    private var metricsReader: TrainingMetricsReader? = null
    private val demoEngine = DemoInferenceEngine()
    private val nativeInference = NativeSflInferenceEngine.createOrNull()
    private val watchMessenger = WatchMessenger(context)
    private val subscription: AutoCloseable
    private val trainingEngine = NativeSflTrainingEngine.createOrNull()
    private var trainingPoll: ScheduledFuture<*>? = null

    var state by mutableStateOf(
        PhoneUiState(
            source = "Waiting for Watch8",
            inferenceRuntimeAvailable = nativeInference != null,
            inferenceConfigured = nativeInference?.configured == true,
            inferenceMessage = if (nativeInference == null) "Native inference runtime is not installed" else "Select an inference deployment config",
        )
    )
        private set

    init {
        if (trainingEngine != null) {
            state = state.copy(training = TrainingStatus(TrainingPhase.IDLE, message = "Ready"))
        }
        subscription = PendingSensorWindows.subscribe { process(it) }
        statusWorker.scheduleWithFixedDelay({
            val endpoint = preferences.getString("aggregationEndpoint", "http://127.0.0.1:50053/status").orEmpty()
            val aggregation = runCatching { readAggregationStatus(endpoint, activeRunId) }
                .getOrElse { JSONObject().put("phase", "OFFLINE").put("message", it.message ?: "Status unavailable") }
            main.post { state = state.copy(aggregation = aggregation) }
        }, 0, 500, TimeUnit.MILLISECONDS)
    }

    fun startTraining(request: TrainingRequest) {
        val training = trainingEngine ?: run {
            state = state.copy(error = "Install the wellbeing_sfl native runtime before starting training")
            return
        }
        state = state.copy(training = TrainingStatus(TrainingPhase.STARTING, message = "Loading model and training assets"), trainingHistory = emptyList(), error = null)
        worker.execute {
            val config = runCatching { JSONObject(File(request.configPath).readText()) }.getOrNull()
            activeRunId = config?.optString("run_id")
            val cut = config?.optJSONObject("training")?.let { if (it.has("cut_layer")) it.getInt("cut_layer") else null }
            main.post { state = state.copy(cuttingLayer = cut) }
            metricsReader = config?.optString("metrics_path")?.takeIf { it.isNotBlank() }?.let {
                val path = File(it).let { file -> if (file.isAbsolute) file else File(File(request.configPath).parentFile, it) }
                TrainingMetricsReader(path, activeRunId.orEmpty())
            }
            runCatching { training.start(request) }
                .onSuccess {
                    main.post { state = state.copy(training = training.status(), error = null) }
                    trainingPoll?.cancel(false)
                    trainingPoll = worker.scheduleWithFixedDelay({
                        runCatching { training.status() }
                            .onSuccess { status ->
                                val recorded = runCatching { metricsReader?.read(status).orEmpty() }.getOrDefault(emptyList())
                                main.post {
                                    val points = recorded + listOfNotNull(status.takeIf { it.loss != null && it.localStep > 0 })
                                    val history = (state.trainingHistory + points).distinctBy { it.round to it.localStep }
                                        .sortedWith(compareBy({ it.round }, { it.localStep }))
                                    state = state.copy(training = status, trainingHistory = history)
                                    if (!status.running) trainingPoll?.cancel(false)
                                }
                            }
                            .onFailure { error -> main.post { state = state.copy(error = error.message) } }
                    }, 1, 1, TimeUnit.SECONDS)
                }
                .onFailure { error -> main.post { state = state.copy(error = error.message, training = TrainingStatus(TrainingPhase.FAILED, message = error.message ?: "Training failed")) } }
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
                            outputs = if (state.source == "Demo inference") emptyList() else state.outputs,
                            source = if (state.source == "Demo inference" || state.outputs.isEmpty()) "Waiting for Watch8" else state.source,
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
        process(SensorWindow(now - 30_000, now, emptyList()), preview = true)
    }

    fun reportError(message: String) { state = state.copy(error = message) }

    fun clearPreview() {
        if (state.source == "Demo inference") state = state.copy(outputs = emptyList(), latestWindow = null, source = "Waiting for Watch8", error = null)
    }

    private fun process(window: SensorWindow, preview: Boolean = false) {
        if (!preview && window.samples.isEmpty()) {
            main.post { reportError("The watch sent an empty sensor window") }
            return
        }
        main.post { state = state.copy(isProcessing = true, error = null,
            outputs = if (preview != (state.source == "Demo inference")) emptyList() else state.outputs,
            latestWindow = window.takeIf { it.samples.isNotEmpty() },
            source = if (preview) "Demo inference" else "Watch8 sensor window") }
        worker.execute {
            val selected = if (preview) {
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
                            source = if (preview) "Demo inference" else "On-device Llama · Watch8 window",
                        )
                        if (!preview) watchMessenger.publish(output)
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
        statusWorker.shutdownNow()
    }
}
