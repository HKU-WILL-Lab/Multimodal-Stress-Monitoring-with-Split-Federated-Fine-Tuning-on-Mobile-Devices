package org.mobihoc.wellbeing.phone.training

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile

/** Consume every completed metrics record, even when several steps finish between UI polls. */
class TrainingMetricsReader(private val file: File, private val runId: String) {
    private var offset = 0L
    fun read(status: TrainingStatus): List<TrainingStatus> {
        if (!file.isFile) return emptyList()
        return RandomAccessFile(file, "r").use { input ->
            if (input.length() < offset) offset = 0
            input.seek(offset)
            buildList {
                while (true) {
                    val start = input.filePointer
                    val line = input.readLine() ?: break
                    val value = try { JSONObject(line) } catch (_: Exception) {
                        input.seek(start)
                        break
                    }
                    offset = input.filePointer
                    if (value.optString("run_id") != runId) continue
                    val loss = value.optDouble("loss", Double.NaN)
                    if (!loss.isFinite()) continue
                    add(status.copy(round = value.getInt("global_round"),
                        localStep = value.getInt("local_step") + 1, loss = loss.toFloat()))
                }
            }
        }
    }
}
