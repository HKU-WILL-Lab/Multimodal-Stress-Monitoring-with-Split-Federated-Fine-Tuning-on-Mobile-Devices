package org.mobihoc.wellbeing.phone.training

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI

/** Poll the coordinator's read-only endpoint; never infer aggregation from client progress. */
fun readAggregationStatus(endpoint: String, expectedRun: String?): JSONObject {
    if (expectedRun.isNullOrBlank()) return JSONObject().put("phase", "NOT_CONNECTED").put("message", "No training session is connected")
    if (endpoint.isBlank()) return JSONObject().put("phase", "UNCONFIGURED")
    val url = URI(endpoint).toURL()
    require(url.protocol == "http" || url.protocol == "https") { "Use an HTTP(S) status URL" }
    val connection = url.openConnection() as HttpURLConnection
    return try {
        connection.connectTimeout = 1500
        connection.readTimeout = 1500
        connection.instanceFollowRedirects = false
        check(connection.responseCode == 200) { "Status endpoint returned ${connection.responseCode}" }
        val value = JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
        require(value.getString("phase") in setOf("IDLE", "WAITING", "AGGREGATING", "COMPLETE", "FAILED")) { "Invalid aggregation status" }
        if (!expectedRun.isNullOrEmpty() && value.optString("runId") != expectedRun) {
            JSONObject().put("phase", "MISMATCH").put("message", "Coordinator is running another session")
        } else value
    } finally { connection.disconnect() }
}
