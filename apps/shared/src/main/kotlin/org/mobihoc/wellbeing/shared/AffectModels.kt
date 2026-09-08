package org.mobihoc.wellbeing.shared

enum class AffectState(val displayName: String, val suggestion: String, val shouldAlert: Boolean) {
    CALM("Calm", "Your current state is steady. Keep it up.", false),
    ENERGIZED("Energized", "You are full of energy. Make the most of it.", false),
    TENSE("Tense", "Take a slow, deep breath.", true),
    LOW_ENERGY("Low Energy", "Take a short rest.", true),
    NEUTRAL_MIXED("Neutral / Mixed", "Your current state is steady.", false),
}

data class ModelOutput(
    val valence: Int,
    val arousal: Int,
    val assessment: String,
    val timestampMillis: Long,
) {
    init {
        require(valence in 1..5) { "valence must be in [1, 5]" }
        require(arousal in 1..5) { "arousal must be in [1, 5]" }
    }

    val state: AffectState = AffectMapper.map(valence, arousal)
}

object AffectMapper {
    fun map(valence: Int, arousal: Int): AffectState = when {
        valence !in 1..5 || arousal !in 1..5 -> AffectState.NEUTRAL_MIXED
        valence == 3 || arousal == 3 -> AffectState.NEUTRAL_MIXED
        valence >= 4 && arousal <= 2 -> AffectState.CALM
        valence >= 4 && arousal >= 4 -> AffectState.ENERGIZED
        valence <= 2 && arousal >= 4 -> AffectState.TENSE
        valence <= 2 && arousal <= 2 -> AffectState.LOW_ENERGY
        else -> AffectState.NEUTRAL_MIXED
    }
}

object TrendFormatter {
    /** The UI keeps at most the latest three 30-second outputs (90 seconds). */
    fun format(outputs: List<ModelOutput>): String {
        val recent = outputs.takeLast(3)
        if (recent.isEmpty()) return "Waiting for the first 30-second window."
        if (recent.size == 1) return "Current state: ${recent.last().state.displayName}."

        val first = recent.first().state
        val last = recent.last().state
        return if (recent.all { it.state == first }) {
            "You have remained ${last.displayName.lowercase()} over the recent ${recent.size * 30} seconds."
        } else {
            "You have gradually moved from ${first.displayName.lowercase()} to ${last.displayName.lowercase()} over the recent ${recent.size * 30} seconds."
        }
    }
}
