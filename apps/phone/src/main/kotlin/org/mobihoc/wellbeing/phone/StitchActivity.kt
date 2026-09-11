package org.mobihoc.wellbeing.phone

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import org.json.JSONArray
import org.json.JSONObject
import org.mobihoc.wellbeing.phone.training.TrainingRequest
import org.mobihoc.wellbeing.shared.SensorKind
import java.io.ByteArrayInputStream

class MainActivity : ComponentActivity() {
    private lateinit var controller: WellbeingController
    private var web: WebView? = null
    private val preferences by lazy { getSharedPreferences("stitch-settings", MODE_PRIVATE) }
    private val origin = "https://appassets.androidplatform.net/stitch/"

    @SuppressLint("SetJavaScriptEnabled", "JavascriptInterface")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        controller = (application as MobiWellbeingApplication).wellbeingController
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (web?.url != origin + "vitals.html") web?.loadUrl(origin + "vitals.html")
                else { isEnabled = false; onBackPressedDispatcher.onBackPressed() }
            }
        })
        setContent {
            val state = controller.state
            AndroidView(modifier = Modifier.fillMaxSize().systemBarsPadding(), factory = { context ->
                WebView(context).apply {
                    web = this
                    setBackgroundColor(0xFFF8F9FF.toInt())
                    settings.javaScriptEnabled = true
                    settings.allowFileAccess = false
                    settings.allowContentAccess = false
                    settings.blockNetworkLoads = true
                    settings.textZoom = 100
                    isVerticalScrollBarEnabled = false
                    addJavascriptInterface(Bridge(), "Wellbeing")
                    webViewClient = object : WebViewClient() {
                        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean =
                            request.url.toString() !in listOf("vitals", "training", "settings").map { origin + it + ".html" }

                        override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse {
                            val url = request.url
                            val path = url.path.orEmpty()
                            if (url.scheme != "https" || url.host != "appassets.androidplatform.net" || !path.startsWith("/stitch/") || path.contains("..")) {
                                return WebResourceResponse("text/plain", "UTF-8", 403, "Forbidden", emptyMap(), ByteArrayInputStream(byteArrayOf()))
                            }
                            return runCatching {
                                val mime = when (path.substringAfterLast('.')) {
                                    "html" -> "text/html"
                                    "css" -> "text/css"
                                    "js" -> "application/javascript"
                                    "woff2" -> "font/woff2"
                                    "ttf" -> "font/ttf"
                                    else -> "application/octet-stream"
                                }
                                WebResourceResponse(mime, "UTF-8", assets.open(path.removePrefix("/")))
                            }.getOrElse {
                                WebResourceResponse("text/plain", "UTF-8", 404, "Not Found", emptyMap(), ByteArrayInputStream(byteArrayOf()))
                            }
                        }
                        override fun onPageFinished(view: WebView, url: String) { render() }
                    }
                    loadUrl(origin + (savedInstanceState?.getString("page") ?: "vitals.html"))
                }
            })
            SideEffect { render(state) }
            DisposableEffect(Unit) { onDispose { web?.removeJavascriptInterface("Wellbeing"); web?.destroy(); web = null } }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString("page", web?.url?.substringAfterLast('/') ?: "vitals.html")
        super.onSaveInstanceState(outState)
    }

    private fun config(): JSONObject {
        val base = getExternalFilesDir(null)!!.absolutePath
        val defaults = JSONObject().apply {
            put("inferenceConfig", "$base/inference/deployment.json")
            put("clientId", "phone-1"); put("configPath", "$base/sfl/config.json")
            put("modelDirectory", "$base/sfl/model"); put("encoderProgram", "$base/sfl/encoder.pte")
            put("datasetPath", "$base/sfl/training.sflsensor")
            put("mainServer", "127.0.0.1:50051"); put("federatedServer", "127.0.0.1:50052")
            put("encoderInputLength", "240")
            put("aggregationEndpoint", "http://127.0.0.1:50053/status")
        }
        defaults.keys().forEach { key -> preferences.getString(key, null)?.let { defaults.put(key, it) } }
        return defaults
    }

    private fun render(state: PhoneUiState = controller.state) {
        val t = state.training
        val current = state.current
        val json = JSONObject().apply {
            put("name", current?.state?.displayName ?: if (state.isProcessing) "Analyzing" else "Collecting")
            put("trend", state.trend)
            put("suggestion", current?.state?.suggestion ?: "Your assessment and suggestion will appear after a sensor window is analyzed.")
            put("assessment", current?.assessment ?: "")
            put("valence", current?.valence ?: JSONObject.NULL); put("arousal", current?.arousal ?: JSONObject.NULL)
            put("processing", state.isProcessing); put("source", state.source); put("error", state.error ?: JSONObject.NULL)
            put("inferenceAvailable", state.inferenceRuntimeAvailable); put("inferenceConfigured", state.inferenceConfigured)
            put("inferenceMessage", state.inferenceMessage); put("config", config())
            put("aggregation", state.aggregation)
            put("sensors", JSONObject().apply {
                mapOf("ACC" to SensorKind.ACCELEROMETER, "PPG" to SensorKind.PPG, "EDA" to SensorKind.EDA, "TEMP" to SensorKind.SKIN_TEMPERATURE).forEach { (label, kind) ->
                    val samples = state.latestWindow?.samples.orEmpty().filter { it.kind == kind }
                    val stride = (samples.size / 160).coerceAtLeast(1)
                    put(label, JSONArray(samples.filterIndexed { i, _ -> i % stride == 0 }.take(161).map { it.values.first().toDouble() }.filter { it.isFinite() }))
                }
            })
            put("training", JSONObject().apply {
                put("cuttingLayer", state.cuttingLayer ?: JSONObject.NULL)
                put("phase", t.phase.name); put("running", t.running)
                put("round", t.round); put("totalRounds", t.totalRounds); put("step", t.localStep); put("total", t.totalLocalSteps)
                put("loss", t.loss?.takeIf { it.isFinite() } ?: JSONObject.NULL); put("message", t.message)
                put("history", JSONArray(state.trainingHistory.map { point -> JSONObject().apply {
                    put("round", point.round); put("step", point.localStep); put("loss", point.loss?.takeIf { it.isFinite() } ?: JSONObject.NULL)
                } }))
            })
        }
        web?.evaluateJavascript("window.renderWellbeing && window.renderWellbeing($json)", null)
    }

    private inner class Bridge {
        @JavascriptInterface fun action(name: String, payload: String) {
            runOnUiThread {
                runCatching {
                    val data = JSONObject(payload)
                    when (name) {
                        "preview" -> controller.simulateNextWindow()
                        "clearPreview" -> controller.clearPreview()
                        "save", "load", "start" -> {
                            val edit = preferences.edit()
                            config().keys().forEach { if (data.has(it)) edit.putString(it, data.getString(it).trim()) }
                            edit.apply()
                            val c = config()
                            if (name == "load") controller.configureInference(c.getString("inferenceConfig"))
                            if (name == "start") {
                                require(!controller.state.training.running) { "A training session is already active" }
                                val length = c.getString("encoderInputLength").toIntOrNull()
                                require(length != null && length > 0) { "Encoder input length must be a positive integer" }
                                controller.startTraining(TrainingRequest(c.getString("clientId"), c.getString("configPath"), c.getString("modelDirectory"),
                                    c.getString("encoderProgram"), c.getString("datasetPath"), c.getString("mainServer"), c.getString("federatedServer"), length))
                            }
                        }
                        "cancel" -> controller.cancelTraining()
                    }
                }.onFailure { controller.reportError(it.message ?: "Invalid settings") }
                render()
            }
        }
    }
}
